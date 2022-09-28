import rdkit.Chem.BRICS as BRICS
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.Descriptors as Descriptors
import re
import warnings
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
#from umap import UMAP
import umap as mp
import time
import trimap
from math import pi
import os

def intersect_MolDBs(db1,db2,simt,fingerprint='RDKIT',output=None,verbose=True):
    db3 = copy.deepcopy(db1)
    keepkeys_db1 = []
    keepkeys_db2 = []
    hitsSMILE = 0
    hitsSimilarity = 0
    for i,k1 in enumerate(db1.dicDB.keys()):
        m1 = db1.dicDB[k1][2]
        SMILE1 = k1
        for k2 in db2.dicDB.keys():
            m2 = db2.dicDB[k2][2]
            SMILE2 = k2
            if SMILE1 == SMILE2:
                hitsSMILE += 1
                if verbose: print(i,'bySMILE',SMILE1,SMILE2)
                keepkeys_db1.append(SMILE1)
                keepkeys_db2.append(SMILE2)
                break
            #if m1.NumAtoms != m2.NumAtoms: continue
            #if m1.NOCount != m2.NOCount: continue
            #if m1.NHOHCount != m2.NHOHCount: continue
            #if m1.RingCount != m2.RingCount: continue
            #if m1.FractionCSP3 != m2.FractionCSP3: continue
            #if m1.NumAliphaticRings != m2.NumAliphaticRings: continue
            #if m1.NumAromaticRings != m2.NumAromaticRings: continue
            similarity = get_MolSimilarity(m1,m2,fingerprint=fingerprint)
            if similarity >= simt:
                hitsSimilarity += 1
                if verbose: print(i,'bySimilarity',SMILE1,SMILE2)
                keepkeys_db1.append(SMILE1)
                keepkeys_db2.append(SMILE2)
                break
    totalhits = hitsSMILE + hitsSimilarity
    sizedb1 = len(db1.dicDB.keys())
    sizedb2 = len(db2.dicDB.keys())
    print('Hits by SMILE: %d'%hitsSMILE)
    print('Hits by Similarity (threshold %.3f): %d'%(simt,hitsSimilarity))
    print('Total hits: %d'%totalhits)
    print('Total db1 = %d, total db2 = %d'%(sizedb1,sizedb2))
    print('Percentage of elements of db1 in db2: %.3f'%(((float(totalhits)/float(sizedb1))*100)))
    print('Percentage of elements of db2 in db1: %.3f'%(((float(totalhits)/float(sizedb2))*100)))
    if output != None:
        for i,k1 in enumerate(keepkeys_db1):
            k2 = keepkeys_db2[i]
            db3.dicDB[k1][0] += ',' + db2.dicDB[k2][0]
            db3.dicDB[k1][1] += ',' + db2.dicDB[k2][1]
        set_keepkeys_db3 = set(keepkeys_db1)
        set_keys_db3 = set(list(db3.dicDB.keys()))
        delkeys_db3 = list(set_keys_db3.difference(set_keepkeys_db3))
        for key in delkeys_db3:
            del db3.dicDB[key]
        db3.print_MolDB(output)
        db3.save_MolDB(output)

def get_MolSimilarity(mol1,mol2,fingerprint='RDKIT',metric='Tanimoto'):
    fp1 = mol1.get_FingerPrint(alg=fingerprint)
    fp2 = mol2.get_FingerPrint(alg=fingerprint)
    if metric == 'Tanimoto':
        return DataStructs.TanimotoSimilarity(fp1,fp2)
    elif  metric == 'Dice':
        return DataStructs.DiceSimilarity(fp1,fp2)
    elif metric == 'Cosine':
        return DataStructs.CosineSimilarity(fp1,fp2)
    elif metric == 'Sokal':
        return DataStructs.SokalSimilarity(fp1,fp2)
    elif metric == 'Russel':
        return DataStructs.RusselSimilarity(fp1,fp2)
    elif metric == 'Kulczynski':
        return DataStructs.KulczynskiSimilarity(fp1,fp2)
    elif metric == 'McConnaughey':
        return DataStructs.McConnaugheySimilarity(fp1,fp2)
    else:
        raise ValueError('Invalid Metric')

class MolDB(object):
    """""
    Class to store a database of molecules/fragments.
    It takes as input:
    - txtDB: file  of format 'SMILE equivalentSMILES IDs'
    - sdfDB: molecular DB in sdf format
    - dicDB: precalculated MolDB object
    - pdbList: 
    - chirality:
    The main attribute of the class is 'dicDB', a dicctionary with the molecules SMILES as keys and
    with lists of form [eqSMILES, IDs, Mol object] as values.
    If paramaters flag is given multiple paramaters of each molecule such as NumAtoms or NOCount are
    calculated and stored.
    """""
    def __init__(self, txtDB = None, dicDB = None, sdfDB = None, pdbList = None, paramaters = False, chirality = True, verbose = True):
        self.txtDB = txtDB
        self.dicDB = dicDB
        self.sdfDB = sdfDB
        self.paramaters = paramaters
        self.chirality = chirality
        if self.txtDB != None and self.dicDB == None and self.sdfDB == None and pdbList == None:
            self.dicDB = {}
            db = open(self.txtDB,'r')
            counteq = 0
            for i,line in enumerate(db):
                line = line.split()
                SMILE = line[0]
                eqSMILES = line[1]
                IDs = line[2]
                mol = Mol(smile=SMILE,allparamaters = self.paramaters, chirality = self.chirality)
                if mol.error == -1: continue
                if SMILE not in self.dicDB:
                    self.dicDB[SMILE] = [eqSMILES,IDs,mol]
                else:
                    counteq+=1
                    old_eqSMILES = self.dicDB[SMILE][0].split(',')
                    new_eqSMILES = eqSMILES.split(',')
                    total_eqSMILES = ','.join(list(set(old_eqSMILES + new_eqSMILES)))
                    self.dicDB[SMILE][1]+=',%s'%IDs
                    self.dicDB[SMILE][0] = total_eqSMILES
                if verbose: print(i+1,IDs,SMILE)
            if verbose: print('Repeated SMILES: %d'%counteq)
        elif self.txtDB == None and self.dicDB != None and self.sdfDB == None and pdbList == None:
            with open(self.dicDB, 'rb') as handle:
                self.dicDB = pickle.load(handle)
        elif self.txtDB == None and self.dicDB == None and self.sdfDB != None and pdbList == None:
            self.dicDB = {}
            DB = Chem.SDMolSupplier(self.sdfDB)
            counteq = 0
            for i,cpd in enumerate(DB):
                mol = Mol(mol2 = cpd, allparamaters = self.paramaters, chirality = self.chirality)
                if mol.error == -1: continue
                SMILE = mol.smile
                eqSMILES = SMILE
                try:
                    IDs = mol.mol.GetProp("_Name")
                except:
                    IDs = 'None'
                if IDs == '' or IDs == 'None':
                    try:
                        IDs = mol.mol.GetProp("Catalog ID")
                    except:
                        pass
                if SMILE not in self.dicDB:
                    self.dicDB[SMILE] = [eqSMILES,IDs,mol]
                else:
                    counteq+=1
                    self.dicDB[SMILE][1]+=',%s'%IDs
                if verbose: print(i+1,IDs,SMILE)
            if verbose: print('Repeated SMILES: %d'%counteq)
        elif self.txtDB == None and self.dicDB == None and self.sdfDB == None and pdbList != None:
            self.dicDB = {}
            counteq = 0
            for i,pdb in enumerate(pdbList):
                mol =  Mol(pdb = pdb, allparamaters = self.paramaters, chirality = self.chirality)
                if mol.error == -1: continue
                SMILE = mol.smile
                eqSMILES = SMILE
                IDs = os.path.basename(pdb).replace(".pdb","")
                if SMILE not in self.dicDB:
                    self.dicDB[SMILE] = [eqSMILES,IDs,mol]
                else:
                    counteq+=1
                    self.dicDB[SMILE][1]+=',%s'%IDs
                if verbose: print(i+1,IDs,SMILE)
            if verbose: print('Unique molecules %d.\nRepeated SMILES: %d'%(len(self.dicDB.keys()),counteq)) 
        else:
            raise KeyError('Provide only a txtDB, a dicDB, or a sdfDB')
        self._get_total_mols()
        self.table = None
    
    def _get_allmols_paramaters(self):
        if self.paramaters: return
        for k in self.dicDB.keys():
            mol = self.dicDB[k][2]
            mol.get_AllParamaters()

    def _get_kmeans(self,n_clusters,data):
        model = KMeans(n_clusters = n_clusters, init = "k-means++")
        labels = model.fit_predict(data)
        centroids = model.cluster_centers_
        clusters = []
        for i in range(n_clusters):
            indexes = np.where(np.array(labels) == i)[0]
            clusters.append(indexes)
        clusters = np.asarray(clusters)
        return labels, centroids, clusters

    def plot_PCA(self, output = None, random_max = None, fpsalg = 'RDKIT', kmeans = False, n_clusters = 1):
        self.get_fingerprints(fpsalg, random_max)
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(self.fingerprints)
        self._plot_reducer(pca_results,output,kmeans,n_clusters)

    def plot_tSNE(self, output = None, random_max = None, fpsalg = 'RDKIT', kmeans = False, n_clusters = 1):
        self.get_fingerprints(fpsalg, random_max)
        tsne = TSNE(n_components=2, verbose = 1, learning_rate='auto',init='pca', n_iter=2500, perplexity=50,metric='hamming')
        tsne_results = tsne.fit_transform(np.asarray(self.fingerprints))
        self._plot_reducer(tsne_results,output,kmeans,n_clusters)

    def plot_UMAP(self, output = None, random_max = None, fpsalg = 'RDKIT', kmeans = False, n_clusters = 1):
        self.get_fingerprints(fpsalg, random_max)
        umap = mp.UMAP(n_neighbors=50, n_epochs=5000, min_dist= 0.5,metric='hamming')
        UMAP_results = umap.fit_transform(self.fingerprints)
        self._plot_reducer(UMAP_results,output,kmeans,n_clusters)

    def plot_trimap(self, output = None, random_max = None, fpsalg = 'RDKIT', kmeans = False, n_clusters = 1):
        self.get_fingerprints(fpsalg, random_max)
        tri = trimap.TRIMAP(n_dims=2, distance='hamming', n_iters = 2500, n_inliers = 30, n_outliers = 5,weight_temp=0.6)
        tri_results = tri.fit_transform(np.asarray(self.fingerprints))
        self._plot_reducer(tri_results,output,kmeans,n_clusters)

    def _plot_reducer(self,reducer_results, output = None, kmeans = False, n_clusters = 1):
        if kmeans:
            labels,centroids,clusters = self._get_kmeans(n_clusters,reducer_results)
            df = pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1],  cluster = labels))
            sns.scatterplot('xaxis', 'yaxis', data=df, hue='cluster',alpha = 0.8, s=15,style='cluster',palette = sns.color_palette("hls", n_clusters))
            plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
        else:
            df = pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1]))
            sns.scatterplot('xaxis', 'yaxis', data=df, alpha = 0.8, s=15)
        if output != None:
            plt.savefig(output+'.png',dpi=300)

    def plot_NPR(self,output=None,zkey=None):
        if isinstance(self.table, pd.DataFrame):
            pass
        else:
            self.get_properties_table()
        
        plt.rcParams['axes.linewidth'] = 1.5
        plt.figure()
        
        if zkey == None:
            ax=sns.scatterplot(x='NPR1',y='NPR2',data=self.table,s=25,linewidth=0.5,alpha=1)
        else:
            x = self.table['NPR1'].tolist()
            y = self.table['NPR2'].tolist()
            z = self.table[zkey].tolist()
            ax= plt.scatter(x=x,y=y,c=z,data=self.table,s=8,linewidth=0.5,alpha=0.75)
            plt.colorbar()
        
        x1, y1 = [0.5, 0], [0.5, 1]
        x2, y2 = [0.5, 1], [0.5, 1]
        x3, y3 = [0,1],[1,1]

        plt.plot(x1, y1,x2,y2,x3,y3,c='gray',ls='--',lw=1)
        plt.xlabel ('NPR1',fontsize=15,fontweight='bold')

        plt.ylabel ('NPR2',fontsize=15,fontweight='bold')

        if zkey == None:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            pass
        
        plt.text(0, 1.01,s='Rod',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')
        plt.text(1, 1.01,s='Sphere',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')
        plt.text(0.5, 0.49,s='Disc',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')

        plt.tick_params ('both',width=2,labelsize=14)
        plt.tight_layout()
        if output != None:
            plt.savefig(output+'.png',dpi=300)

    def plot_paramaters_PCA(self,output=None):
        if isinstance(self.table, pd.DataFrame):
            pass
        else:
            self.get_properties_table()

        descriptors = self.table[['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3', 'TPSA','RadiusOfGyration']].values
        descriptors_std = StandardScaler().fit_transform(descriptors)
        pca = PCA()
        descriptors_2d = pca.fit_transform(descriptors_std)
        descriptors_pca= pd.DataFrame(descriptors_2d)
        descriptors_pca.index = self.table.index
        descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
        descriptors_pca.head(5)

        print(pca.explained_variance_ratio_) #Let's plot PC1 vs PC2
        print(sum(pca.explained_variance_ratio_))

        scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
        scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

        descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
        descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

        plt.rcParams['axes.linewidth'] = 1.5
        plt.figure(figsize=(6,6))

        ax=sns.scatterplot(x='PC1_normalized',y='PC2_normalized',data=descriptors_pca,s=20,palette=sns.color_palette("Set2", 3),linewidth=0.2,alpha=1)

        plt.xlabel ('PC1',fontsize=20,fontweight='bold')
        ax.xaxis.set_label_coords(0.98, 0.45)
        plt.ylabel ('PC2',fontsize=20,fontweight='bold')
        ax.yaxis.set_label_coords(0.45, 0.98)

        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        lab=['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3','TPSA','RadiusOfGyration']

        l=np.transpose(pca.components_[0:2, :])
        print(lab)
        print(l)
        print(np.linalg.norm(l,axis=1))
        
        n = l.shape[0]
        for i in range(n):
            plt.arrow(0, 0, l[i,0], l[i,1],color= 'k',alpha=0.5,linewidth=1.8,head_width=0.025)
            plt.text(l[i,0]*1.25, l[i,1]*1.25, lab[i], color = 'k',va = 'center', ha = 'center',fontsize=16)
        
        circle = plt.Circle((0,0), 1, color='gray', fill=False,clip_on=True,linewidth=1.5,linestyle='--')
        plt.tick_params ('both',width=2,labelsize=18)

        ax.add_artist(circle)
        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.tight_layout()
        
        if output != None:
            plt.savefig(output+'.png',dpi=300)

    def plot_radar(self,output=None):
        if isinstance(self.table, pd.DataFrame):
            pass
        else:
            self.get_properties_table()

        data=pd.DataFrame()

        data['MolWt']=[i/500 for i in self.table['MolWt']]
        data['LogP']=[i/5 for i in self.table['LogP']]
        data['nHA']=[i/10 for i in self.table['NumHAcceptors']]
        data['nHD']=[i/3 for i in self.table['NumHDonors']]
        data['nRotB']=[i/10 for i in self.table['NumRotatableBonds']]
        data['TPSA']=[i/140 for i in self.table['TPSA']]
        
        categories=list(data.columns) 
        N = len(categories)
        values=data[categories].values[0]
        values=np.append(values,values[:1])
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        Ro5_up=[1,1,1,1,1,1,1] #The upper limit for bRo5
        Ro5_low=[0.5,0.1,0,0.25,0.1,0.5,0.5]  #The lower limit for bRo5
    
        fig = plt.figure()
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        plt.xticks(angles[:-1], categories,color='k',size=15,ha='center',va='top',fontweight='book')

        plt.tick_params(axis='y',width=4,labelsize=6, grid_alpha=0.05)

        ax.set_rlabel_position(0)

        all_values = []
        for i in data.index:
            values=data[categories].values[i]
            values=np.append(values,values[:1])
            all_values.append(values)
            plt.plot(angles, values, linewidth=0.6 ,color='steelblue',alpha=0.5)

        all_values = np.asarray(all_values)
        average_values  = np.mean(all_values,axis=0)
        plt.plot(angles, average_values, linewidth=1 ,linestyle='-',color='orange')

        ax.grid(axis='y',linewidth=1.5,linestyle='dotted',alpha=0.8)
        ax.grid(axis='x',linewidth=2,linestyle='-',alpha=1)
        
        plt.plot(angles, Ro5_up, linewidth=2, linestyle='-',color='red')
        plt.plot(angles, Ro5_low, linewidth=2, linestyle='-',color='red')
        
        if output != None:
            plt.savefig(output,dpi=300)

    def save_MolDB(self,output):
        with open(output+'.p', 'wb') as handle:
            pickle.dump(self.dicDB, handle)

    def print_MolDB(self,output):
        f = open(output+'.txt','w')
        for k in self.dicDB.keys():
            f.write(k + " " + str(self.dicDB[k][0]) + " " + str(self.dicDB[k][1]) + '\n')
        f.close()

    def save_MolDB_sdf(self,output):
        with Chem.SDWriter(output) as w:
            for k in self.dicDB.keys():
                mol = self.dicDB[k][2].mol
                ID = self.dicDB[k][1]
                mol.SetProp("_Name",ID)
                w.write(mol,)

    def get_fingerprints(self, alg='RDKIT', random_max=None):
        fps = []
        if random_max == None:
            keys = self.dicDB.keys()
            total = self.size
        elif random_max > self.size:
            keys = self.dicDB.keys()
            total = self.size
        else:
            keys = random.sample(list(self.dicDB.keys()), random_max)
            total = len(keys)
        for i,k in enumerate(keys):
            print(alg + ': ' + str(i) + '/' + str(total))
            mol = self.dicDB[k][2]
            molfp = mol.get_FingerPrint(alg)
            molfp = np.asarray(list((molfp.ToBitString())))
            fps.append(molfp)
        self.fingerprints = fps
        return fps 

    def _get_total_mols(self):
        self.size = len(self.dicDB.keys())
        
    def _remove_anchorings(self):
        kekuleerror1 = 0
        kekuleerror2 = 0
        SMILES = list(self.dicDB.keys())
        totalsmiles = len(SMILES)
        for i,SMILE in enumerate(SMILES):
            print(str(i+1) + "/" + str(totalsmiles))
            eqSMILES = self.dicDB[SMILE][0].split(',')
            IDs = self.dicDB[SMILE][1]
            #Check SMILE
            new_SMILE = ''
            auxmol = Mol(smile=SMILE)
            auxerror = auxmol._remove_anchorings()
            
            #Check eqSMILES and define new_eqSMILES
            new_eqSMILES = []
            for eqSMILE in eqSMILES:
                auxmol2 = Mol(smile=eqSMILE)
                auxerror2 = auxmol2._remove_anchorings()
                if auxerror2:
                    new_eqSMILES.append(auxmol2.smile)
            
            #Define new_SMILE and count errors
            if auxerror:
                new_SMILE = auxmol.smile
            else:
                if len(new_eqSMILES) > 1:
                    new_SMILE = new_eqSMILES[0]
                    kekuleerror2+=1
                else:
                    kekuleerror1+=1
            
            #Modify dicDB
            del self.dicDB[SMILE]
            if new_SMILE != '':
                new_eqSMILES = ','.join(new_eqSMILES)
                self.dicDB[new_SMILE] = [new_eqSMILES,IDs,auxmol]

            i+=1
            print('-----------------------------------------------')
        print('Total analysed smiles: %d'%totalsmiles)
        print('Total modified smiles: %d'%(totalsmiles-kekuleerror1-kekuleerror2))
        print('SMILES with kekule error substituted by an eqSMILE: %d'%kekuleerror2)
        print('SMILES with a kekule error without an eqSMULE: %d'%kekuleerror1)

    def get_properties_table(self):
        if self.paramaters: return
        else:
            self._get_allmols_paramaters()
            self.paramaters = True

        table = pd.DataFrame()
        for i,k in enumerate(self.dicDB.keys()):
            mol = self.dicDB[k][2]
            table.loc[i,'id'] = self.dicDB[k][1]
            table.loc[i,'smile']=k
            try:
                table.loc[i,'IC50'] = float(mol.mol.GetProp('Value'))/1000000000
                table.loc[i,'pIC50'] = -np.log10(float(mol.mol.GetProp('Value'))/1000000000)
            except:
                pass
            table.loc[i,'MolWt']=mol.MolWt
            table.loc[i,'LogP']=mol.LogP
            table.loc[i,'NumHAcceptors']=mol.NumHAcceptors
            table.loc[i,'NumHDonors']=mol.NumHDonors
            table.loc[i,'NumHeteroatoms']=mol.NumHeteroatoms
            table.loc[i,'NumRotatableBonds']=mol.NumRotatableBonds
            table.loc[i,'NumHeavyAtoms']=mol.NumHeavyAtoms
            table.loc[i,'NumAliphaticCarbocycles']=mol.NumAliphaticCarbocycles
            table.loc[i,'NumAliphaticHeterocycles']=mol.NumAliphaticHeterocycles
            table.loc[i,'NumAliphaticRings']=mol.NumAliphaticRings
            table.loc[i,'NumAromaticCarbocycles']=mol.NumAromaticCarbocycles
            table.loc[i,'NumAromaticHeterocycles']=mol.NumAromaticHeterocycles
            table.loc[i,'NumAromaticRings']=mol.NumAromaticRings
            table.loc[i,'RingCount']=mol.RingCount
            table.loc[i,'FractionCSP3']=mol.FractionCSP3
            table.loc[i,'TPSA']=mol.TPSA
            table.loc[i,'NPR1']=mol.NPR1
            table.loc[i,'NPR2']=mol.NPR2
            table.loc[i,'InertialShapeFactor']=mol.InertialShapeFactor
            table.loc[i,'RadiusOfGyration']=mol.RadiusOfGyration
        self.table = table

    def filter_similarity(self,simthreshold=1,fingerprint='RDKIT',verbose=True):
        if not self.paramaters:
            self._get_allmols_paramaters()
            self.paramaters = True

        new_dicDB = {}
        count = 0
        for i,key1 in enumerate(self.dicDB.keys()):
            count+=1
            SMILE1 = key1
            eqSMILES1 = self.dicDB[key1][0]
            IDs1 = self.dicDB[key1][1]
            mol1 = self.dicDB[key1][2]
            if len(new_dicDB.keys()) == 0:
                new_dicDB[SMILE1] = [eqSMILES1,IDs1,mol1]
            for j,key2 in enumerate(new_dicDB.keys()):
                SMILE2 = key2
                eqSMILES2 = new_dicDB[key2][0]
                IDs2 = new_dicDB[key2][1]
                mol2 = new_dicDB[key2][2]
                if mol1.NumAtoms != mol2.NumAtoms or mol1.NOCount != mol2.NOCount or mol1.RingCount != mol2.RingCount or mol1.NHOHCount != mol2.NHOHCount or mol1.FractionCSP3 != mol2.FractionCSP3 or mol1.NumAliphaticRings != mol2.NumAliphaticRings or  mol1.NumAromaticRings != mol2.NumAromaticRings:
                    if j == len(new_dicDB.keys())-1:
                        new_dicDB[SMILE1] = [eqSMILES1,IDs1,mol1]
                        break
                    else:
                        continue
                else:
                    similarity = get_MolSimilarity(mol1,mol2,fingerprint=fingerprint)
                    if j == len(new_dicDB.keys())-1 and similarity >= simthreshold:
                        if verbose: print("0:" + new_dicDB[SMILE2][0] + " " + SMILE1)
                        _eqSMILES1 = eqSMILES1.split(',')
                        _eqSMILES2 = eqSMILES2.split(',')
                        new_eqSMILES = ','.join(list(set(_eqSMILES1 + _eqSMILES2)))
                        new_dicDB[SMILE2][0] = new_eqSMILES
                        _IDs1 = IDs1.split(',')
                        _IDs2 = IDs2.split(',')
                        new_IDs = ','.join(list(set(_IDs1 + _IDs2)))
                        new_dicDB[SMILE2][1] = new_IDs
                    elif similarity >= simthreshold:
                        if verbose: print("1:" + new_dicDB[SMILE2][0] + " " + SMILE1)
                        _eqSMILES1 = eqSMILES1.split(',')
                        _eqSMILES2 = eqSMILES2.split(',')
                        new_eqSMILES = ','.join(list(set(_eqSMILES1 + _eqSMILES2)))
                        new_dicDB[SMILE2][0] = new_eqSMILES
                        _IDs1 = IDs1.split(',')
                        _IDs2 = IDs2.split(',')
                        new_IDs = ','.join(list(set(_IDs1 + _IDs2)))
                        new_dicDB[SMILE2][1] = new_IDs
                        break
                    else:
                        continue
            if verbose:
                print('Unique molecules after filtering: %d'%len(new_dicDB.keys()))
                print('Analized molecules: %d'%count)
            else:
                if count % 500 == 0:  print('Analized compounds: %d'%count)
        self.dicDB = new_dicDB
        self._get_total_mols()


class Mol(object):
    """"""
    """"""
    def __init__(self ,smile = None, InChI = None, mol2 = None, pdb= None,allparamaters = False, chirality = True):
        if smile != None and InChI == None and mol2 == None and pdb == None:
            self.smile = smile
            if not chirality:
                self.smile = self.smile.replace("@","")
            self.mol = Chem.MolFromSmiles(self.smile)
            if self.mol == None:
                self.error = -1
            else:
                self.error = 0
        elif smile == None and InChI != None and mol2 == None and pdb == None:
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
            if self.mol == None:
                self.error = -1
            else:
                self.error = 0
            self.smile = Chem.MolToSmiles(self.mol)
            if not chirality:
                self.smile = self.smile.replace("@","")
        elif smile == None and InChI == None and mol2 != None and pdb == None:
            self.mol = mol2
            if self.mol == None:
                self.error = -1
            else:
                self.error = 0
            self.smile = Chem.MolToSmiles(self.mol)
            if not chirality:
                self.smile = self.smile.replace("@","")
        elif smile == None and InChI == None and mol2 == None and pdb != None:
            self.mol = Chem.MolFromPDBFile(pdb)
            if self.mol == None:
                self.error = -1
            else:
                self.error = 0
            self.smile = Chem.MolToSmiles(self.mol)
            if not chirality:
                self.smile = self.smile.replace("@","")
        else:
            warnings.warn(f'Provide only a smile, a InchI, a mol2 or a pdb')
        if allparamaters:
            self.get_AllParamaters()

    def get_BRICSdecomposition(self,clean=True):
        self.fragments = list(BRICS.BRICSDecompose(self.mol))
        if clean:
            self.fragments = [re.sub("(\[.*?\])", "[*]", frag) for frag in self.fragments]

    def get_AllParamaters(self):
        try:
            Chem.SanitizeMol(self.mol)
        except:
            self.error = -1
            return
        self.get_NumAtoms()
        self.get_NOCount()
        self.get_NHOHCount()
        self.get_RingCount()
        self.get_sp3()
        self.get_NumAliphaticRings()
        self.get_NumAromaticRings()
        self.get_MolWt()
        self.get_LogP()
        self.get_NumHAcceptors()
        self.get_NumHDonors()
        self.get_NumHeteroatoms()
        self.get_NumRotatableBonds()
        self.get_NumHeavyAtoms()
        self.get_NumAliphaticCarbocycles()
        self.get_NumAliphaticHeterocycles()
        self.get_NumAromaticCarbocycles()
        self.get_NumAromaticHeterocycles()
        self.get_TPSA()
        self.get_NPR1()
        self.get_NPR2()
        self.get_InertialShapeFactor()
        self.get_RadiusOfGyration()

    def _remove_anchorings(self):
        print('Old SMILE: ' + self.smile)
        old_smile = self.smile
        #Get indeces of the anchoring points in the smile:
        count = 0
        new_smile = old_smile
        while new_smile.startswith('[*].'): new_smile = new_smile[4:]
        while new_smile.endswith('.[*]'): new_smile = new_smile[:-4]
        indices = [i for i, c in enumerate(new_smile) if c == '*']
        new_smile = list(new_smile)
        for atom in self.mol.GetAtoms():
            if atom.GetSymbol() == '*' and count < len(indices):
                valence = atom.GetExplicitValence()
                if valence == 0:
                    new_smile[indices[count]] = ''
                    new_smile[indices[count]-1] = ''
                    new_smile[indices[count]+1] = ''
                elif valence == 1:
                    if atom.GetIsAromatic():
                        new_smile[indices[count]] = 'h'
                    else:
                        new_smile[indices[count]] = 'H'
                    count+=1
#                elif valence == 2:
#                    if atom.GetIsAromatic():
#                        new_smile[indices[count]] = 'o'
#                    else:
#                        new_smile[indices[count]] = 'O'
#                    count+=1
#                elif valence == 3:
#                    if atom.GetIsAromatic():
#                        new_smile[indices[count]] = 'n'
#                    else:
#                        new_smile[indices[count]] = 'N'
#                    count+=1
#                elif valence == 4:
                elif valence > 1 and valence < 5:
                    if atom.GetIsAromatic():
                        new_smile[indices[count]] = 'c'
                    else:
                        new_smile[indices[count]] = 'C'
                    count+=1
                elif valence == 5:
                    if atom.GetIsAromatic():
                        new_smile[indices[count]] = 'p'
                    else:
                        new_smile[indices[count]] = 'P'
                    count+=1
                elif valence == 6:
                    if atom.GetIsAromatic():
                        new_smile[indices[count]] = 's'
                    else:
                        new_smile[indices[count]] = 'S'
                    count+=1
                elif valence > 6:
                    return False
                else:
                    raise ValueError('The anchoring point %d (*) of %s have a valence %d greater than 4. %s'%(count,old_smile,valence,''.join(new_smile)))
        new_smile = ''.join(new_smile)
        self.smile = new_smile
        try:
            self.mol = Chem.MolFromSmiles(self.smile)
            if self.mol == None:
                print('Kekulize ERROR')
                return False
            else:
                print('New Smile: ' + self.smile)
                return True
        except:
            print('Kekulize ERROR')
            return False

    def get_NumAtoms(self):
        try:
            self.NumAtoms = self.mol.GetNumAtoms()
        except:
            self.NumAtoms = None

    def get_NOCount(self):
        try:
            self.NOCount = Lipinski.NOCount(self.mol)
        except:
            self.NOCount = None

    def get_NHOHCount(self):
        try: self.NHOHCount = Lipinski.NHOHCount(self.mol)
        except: self.NHOHCount = None

    def get_RingCount(self):
        try: self.RingCount = Lipinski.RingCount(self.mol)
        except: None

    def get_sp3(self):
        try: self.FractionCSP3 = Lipinski.FractionCSP3(self.mol)
        except: None

    def get_NumAliphaticRings(self):
        try: self.NumAliphaticRings = Lipinski.NumAliphaticRings(self.mol)
        except: None

    def get_NumAromaticRings(self):
        try: self.NumAromaticRings = Lipinski.NumAromaticRings(self.mol)
        except: None

    def get_MolWt(self):
        try: self.MolWt = Descriptors.ExactMolWt(self.mol)
        except: None

    def get_LogP(self):
        try: self.LogP = Chem.Descriptors.MolLogP(self.mol)
        except: None

    def get_NumHAcceptors(self):
        try: self.NumHAcceptors = Chem.Descriptors.NumHAcceptors(self.mol)
        except: None

    def get_NumHDonors(self):
        try: self.NumHDonors = Chem.Descriptors.NumHDonors(self.mol)
        except: None

    def get_NumHeteroatoms(self):
        try: self.NumHeteroatoms = Chem.Descriptors.NumHeteroatoms(self.mol)
        except: None

    def get_NumRotatableBonds(self):
        try: self.NumRotatableBonds = Chem.Descriptors.NumRotatableBonds(self.mol)
        except: None

    def get_NumHeavyAtoms(self):
        try: self.NumHeavyAtoms = Chem.Descriptors.HeavyAtomCount(self.mol)
        except: None

    def get_NumAliphaticCarbocycles(self):
        try: self.NumAliphaticCarbocycles = Chem.Descriptors.NumAliphaticCarbocycles(self.mol)
        except: None

    def get_NumAliphaticHeterocycles(self):
        try: self.NumAliphaticHeterocycles = Chem.Descriptors.NumAliphaticHeterocycles(self.mol)
        except: None

    def get_NumAromaticCarbocycles(self):
        try: self.NumAromaticCarbocycles = Chem.Descriptors.NumAromaticCarbocycles(self.mol)
        except: None

    def get_NumAromaticHeterocycles(self):
        try: self.NumAromaticHeterocycles = Chem.Descriptors.NumAromaticHeterocycles(self.mol)
        except: None

    def get_TPSA(self):
        try: self.TPSA = Chem.Descriptors.TPSA(self.mol)
        except: None

    def get_NPR1(self):
        try: self.NPR1 = Chem.rdMolDescriptors.CalcNPR1(self.mol)
        except: None

    def get_NPR2(self):
        try: self.NPR2 = Chem.rdMolDescriptors.CalcNPR2(self.mol)
        except: None

    def get_InertialShapeFactor(self):
        try: self.InertialShapeFactor = Chem.Descriptors3D.InertialShapeFactor(self.mol)
        except: None

    def get_RadiusOfGyration(self):
        try: self.RadiusOfGyration = Chem.Descriptors3D.RadiusOfGyration(self.mol)
        except: None

    def get_FingerPrint(self,alg='RDKIT'):
        if alg == 'RDKIT':
            self.FingerPrint = Chem.RDKFingerprint(self.mol)
        elif alg == 'Morgan2':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 1, nBits=2048)
        elif alg == 'Morgan4':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
        elif alg == 'Morgan6':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 3, nBits=2048)
        elif alg == 'Morgan8':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 4, nBits=2048)
        else:
            raise ValueError('Invalid fingerprint algorithm')
        return self.FingerPrint
