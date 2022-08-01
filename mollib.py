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
import matplotlib.pyplot as plt
import seaborn as sns
import random
from umap import UMAP
import time
import trimap

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
            if m1.NumAtoms != m2.NumAtoms: continue
            if m1.NOCount != m2.NOCount: continue
            if m1.NHOHCount != m2.NHOHCount: continue
            if m1.rings != m2.rings: continue
            if m1.sp3 != m2.sp3: continue
            if m1.NumAliphaticRings != m2.NumAliphaticRings: continue
            if m1.NumAromaticRings != m2.NumAromaticRings: continue
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

def filter_db_similarity(indb,outdb,fingerprint='RDKIT',verbose=True):
    inp = open(indb,'r')
    out = open(outdb,'w')
    uniquefrags = {}
    count = 0
    for i, line in enumerate(inp):
        count+=1
        line = line.split()
        SMILE = line[0]
        IDs = line[-1]
        m1 = Mol(smile=SMILE)
        m1_atoms = m1.mol.GetNumAtoms()
        m1_NOCount = Lipinski.NOCount(m1.mol)
        m1_NHOHCount = Lipinski.NHOHCount(m1.mol)
        m1_rings = Lipinski.RingCount(m1.mol)
        m1_sp3 = Lipinski.FractionCSP3(m1.mol)
        m1_NumAliphaticRings = Lipinski.NumAliphaticRings(m1.mol)
        m1_NumAromaticRings = Lipinski.NumAromaticRings(m1.mol)
        if len(uniquefrags.keys()) == 0:
            uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOCount,m1_NHOHCount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
        for j,k in enumerate(uniquefrags.keys()):
            m2 = uniquefrags[k][2]
            m2_atoms = uniquefrags[k][3]
            m2_NOCount = uniquefrags[k][4]
            m2_NHOHCount = uniquefrags[k][5]
            m2_rings = uniquefrags[k][6]
            m2_sp3 = uniquefrags[k][7]
            m2_NumAliphaticRings = uniquefrags[k][8]
            m2_NumAromaticRings = uniquefrags[k][9]
            if m1_atoms != m2_atoms or m1_NOCount != m2_NOCount or m1_rings != m2_rings or m1_NHOHCount != m2_NHOHCount or m1_sp3 != m2_sp3 or m1_NumAliphaticRings != m2_NumAliphaticRings or  m1_NumAromaticRings != m2_NumAromaticRings:
                if j == len(uniquefrags.keys())-1:
                    uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOCount,m1_NHOHCount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                    break
                else:
                    continue
            else:
                similarity =  get_MolSimilarity(m1,m2,fingerprint=fingerprint)
                if j == len(uniquefrags.keys())-1 and similarity != 1:
                    uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOCount,m1_NHOHCount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                    break
                elif j == len(uniquefrags.keys())-1 and similarity == 1:
                    print("0: " + uniquefrags[k][0] + " " +SMILE)
                    uniquefrags[k][0] += ',' + SMILE
                    uniquefrags[k][1] += ',' + IDs
                    break
                elif similarity == 1:
                    print("1: " + uniquefrags[k][0] + " " +SMILE)
                    uniquefrags[k][0] += ',' + SMILE
                    uniquefrags[k][1] += ',' + IDs
                    break
                else:
                    continue
        if verbose:
            print('Unique fragments: %d'%len(uniquefrags.keys()))
            print('Analized fragments: %d'%count)
        else:
            if count % 1000 == 0:  print('Analized fragments: %d'%count)

    inp.close()
    if verbose: print("Saving into file")
    for k in uniquefrags.keys():
        out.write(k + " " + uniquefrags[k][0] + " " +uniquefrags[k][1] + "\n")
    out.close()

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
    The main attribute of the class is 'dicDB', a dicctionary with the molecules SMILES as keys and
    with lists of form [eqSMILES, IDs, Mol object] as values.
    If paramaters flag is given multiple paramaters of each molecule such as NumAtoms or NOCount are
    calculated and stored.
    """""
    def __init__(self, txtDB = None, dicDB = None, sdfDB = None, paramaters = False, verbose = False):
        self.txtDB = txtDB
        self.dicDB = dicDB
        self.sdfDB = sdfDB
        if self.txtDB != None and self.dicDB == None and self.sdfDB == None:
            self.dicDB = {}
            db = open(self.txtDB,'r')
            for i,line in enumerate(db):
                line = line.split()
                SMILE = line[0]
                eqSMILES = line[1]
                IDs = line[2]
                m1 = Mol(smile=SMILE,allparamaters = paramaters)
                if verbose: print(i+1,SMILE)
                self.dicDB[SMILE] = [eqSMILES,IDs,m1]
        elif self.txtDB == None and self.dicDB != None and self.sdfDB == None:
            with open(self.dicDB, 'rb') as handle:
                self.dicDB = pickle.load(handle)
        elif self.txtDB == None and self.dicDB == None and self.sdfDB != None:
            self.dicDB = {}
            DB = Chem.SDMolSupplier(self.sdfDB)
            for i,cpd in enumerate(DB):
                m1 = Mol(mol2 = cpd, allparamaters = paramaters)
                SMILE = m1.smile
                eqSMILES = None
                IDs = None
                if verbose: print(i+1,SMILE)
                self.dicDB[SMILE] = [eqSMILES,IDs,m1]
                #print(self.dicDB[SMILE][-1].mol.GetProp('Value'))
        else:
            raise KeyError('Provide only a txtDB, a dicDB, or a sdfDB')
        self._get_total_mols()

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
        umap = UMAP(n_neighbors=50, n_epochs=5000, min_dist= 0.5,metric='hamming')
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

    def save_MolDB(self,output):
        with open(output+'.p', 'wb') as handle:
            pickle.dump(self.dicDB, handle)

    def print_MolDB(self,output):
        f = open(output+'.txt','w')
        for k in self.dicDB.keys():
            f.write(k + " " + str(self.dicDB[k][0]) + " " + str(self.dicDB[k][1]) + '\n')
        f.close()

    def get_fingerprints(self, alg='RDKIT', random_max=None):
        fps = []
        if random_max == None:
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

class Mol(object):
    """"""
    """"""
    def __init__(self ,smile = None, InChI = None, mol2 = None, allparamaters = False):
        if smile != None and InChI == None and mol2 == None:
            self.smile = smile
            self.mol = Chem.MolFromSmiles(self.smile)
        elif smile == None and InChI != None and mol2 == None:
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
            self.smile = Chem.MoltToSmiles(self.mol)
        elif smile == None and InChI == None and mol2 != None:
            self.mol = mol2
            self.smile = Chem.MolToSmiles(self.mol)
        else:
            warnings.warn(f'Provide only a smile, a InchI or a mol2 RDKIT object')
        if allparamaters:
            self.get_AllParamaters()

    def get_BRICSdecomposition(self):
        self.fragments = list(BRICS.BRICSDecompose(self.mol))

    def get_clean_fragments(self):
        if self.fragments == None: self.get_BRICSdecomposition()
        self.cfragments = [re.sub("(\[.*?\])", "[*]", frag) for frag in self.fragments]
   
    def get_AllParamaters(self):
        self.get_NumAtoms()
        self.get_NOCount()
        self.get_NHOHCount()
        self.get_rings()
        self.get_sp3()
        self.get_NumAliphaticRings()
        self.get_NumAromaticRings()
        self.get_MolWt()

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
        self.NumAtoms = self.mol.GetNumAtoms()
        return self.NumAtoms

    def get_NOCount(self):
        self.NOCount = Lipinski.NOCount(self.mol)
        return self.NOCount

    def get_NHOHCount(self):
        self.NHOHCount = Lipinski.NHOHCount(self.mol)
        return self.NHOHCount

    def get_rings(self):
        self.rings = Lipinski.RingCount(self.mol)
        return self.rings

    def get_sp3(self):
        self.sp3 = Lipinski.FractionCSP3(self.mol)
        return self.sp3

    def get_NumAliphaticRings(self):
        self.NumAliphaticRings = Lipinski.NumAliphaticRings(self.mol)
        return self.NumAliphaticRings

    def get_NumAromaticRings(self):
        self.NumAromaticRings = Lipinski.NumAromaticRings(self.mol)
        return self.NumAromaticRings

    def get_MolWt(self):
        self.MolWt = Descriptors.ExactMolWt(self.mol)
        return self.MolWt

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
