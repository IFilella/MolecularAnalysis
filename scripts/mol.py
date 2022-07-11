import rdkit.Chem.BRICS as BRICS
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.Lipinski as Lipinski
import re
import warnings
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random

def intersect_molDBs(db1,db2,simt,output=None,verbose=True):
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
            similarity = get_MolSimilarity(m1,m2)
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
        for i,k1 in keepkeys_db1:
            k2 = keepkeys_db2[i]
            db3.dicDB[k1][0] += ',' + db2.dicDB[k2][0]
            db3.dicDB[k1][1] += ',' + db2.dicDB[k2][1]
        set_keepkeys_db3 = set(keepkeys_db1)
        set_keys_db3 = set(list(db3.dicDB.keys()))
        delkeys_db3 = list(set_keys_db3.difference(set_keepkeys_db3))
        for key in delkeys_db3:
            del db3.dicDB[key]
        db3.print_molDB(output+'.txt')
        db3.save_molDB(output+'.p')

def filter_db_similarity(indb,outdb,verbose=True):
    inp = open(indb,'r')
    out = open(outdb,'w')
    uniquefrags = {}
    count = 0
    for i, line in enumerate(inp):
        count+=1
        line = line.split()
        SMILE = line[0]
        IDs = line[-1]
        m1 = mol(smile=SMILE)
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
                similarity =  get_MolSimilarity(m1,m2)
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

def plot_TSNE(dbs,names,output=None, random_max = 5000, delimiter = None):
    X = []
    Y = []
    for i,db in enumerate(dbs):
        if delimiter == None:
            name = names[i]
        else:
            name = names[i].split(delimiter)[0]
        print(name)
        fps = db.get_fingerprints(random_max)
        X.extend(fps)
        Y.extend([name]*len(fps))
    X = np.asarray(X)
    print('Computing TSNE')
    tsne_results = TSNE(n_components=2, verbose = 1, learning_rate='auto',init='pca').fit_transform(X)
    df = pd.DataFrame(dict(xaxis=tsne_results[:,0], yaxis=tsne_results[:,1],  molDB = Y))
    plt.figure()
    sns.scatterplot('xaxis', 'yaxis', data=df, hue='molDB',alpha = 0.5, s=3,style='molDB')
    if output != None:
        plt.savefig(output+'.png')
    plt.show()

def plot_PCA(dbs,names,output=None, random_max = 5000, delimiter = None):
    X = []
    Y = []
    for i,db in enumerate(dbs):
        if delimiter == None:
            name = names[i]
        else:
            name = names[i].split(delimiter)[0]
        print(name)
        fps = db.get_fingerprints(random_max)
        X.extend(fps)
        Y.extend([name]*len(fps))
    X = np.asarray(X)
    print('Computing PCA')
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    df = pd.DataFrame(dict(pca1st=pca_results[:,0], pca2nd=pca_results[:,1],  molDB = Y))
    plt.figure()
    sns.scatterplot('pca1st', 'pca2nd', data=df, hue='molDB',alpha = 0.5, s=3,style='molDB')
    if output != None:
        plt.savefig(output+'.png')
    plt.show()

def read_compoundDB(data):
    compoundDB = Chem.SDMolSupplier(data)
    return compoundDB


def get_MolSimilarity(mol1,mol2,metric='Tanimoto'):
    fp1 = Chem.RDKFingerprint(mol1.mol)
    fp2 = Chem.RDKFingerprint(mol2.mol)
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

def get_no_anchoring_frag(frag):
    noanfrag = re.sub("\[.*?\]", "", frag)
    noanfrag = re.sub("\(\)","",noanfrag)
    return noanfrag

class molDB(object):
    """""
    Object containing a database of molecules/fragments.
    It reads a file 'txtDB' of format (SMILE equivalentSMILES IDs) and transform it into a dicc 'dicDB'
    which can also contain physical paramaters of the molecules/fragments if paramaters flag is given.
    It can also load a precalculated 'dicDB' molDB object.
    """""
    def __init__(self, txtDB = None, dicDB = None, paramaters = False, verbose = False):
        self.txtDB = txtDB
        self.dicDB = dicDB
        if self.txtDB != None and self.dicDB == None:
            self.dicDB = {}
            db = open(self.txtDB,'r')
            for i,line in enumerate(db):
                line = line.split()
                SMILE = line[0]
                eqSMILES = line[1]
                IDs = line[2]
                m1 = mol(smile=SMILE,allparamaters = paramaters)
                if verbose: print(i+1,SMILE)
                self.dicDB[SMILE] = [eqSMILES,IDs,m1]
        elif self.txtDB == None and self.dicDB != None:
            with open(self.dicDB, 'rb') as handle:
                self.dicDB = pickle.load(handle)
        elif self.txtDB == None and self.dicDB == None:
            raise KeyError('Either a txtDB or a dicDB must be provided')
        else:
            raise KeyError('Provide only a txtDB or a dicDB, not both simultaniusly')
        self._get_total_mols()
            
    def save_molDB(self,output):
        with open(output+'.p', 'wb') as handle:
            pickle.dump(self.dicDB, handle)

    def print_molDB(self,output):
        f = open(output+'.txt','w')
        for k in self.dicDB.keys():
            f.write(k + " " + self.dicDB[k][0] + " " + self.dicDB[k][1] + '\n')
        f.close()

    def get_fingerprints(self,random_max=None):
        fps = []
        if random_max == None:
            keys = self.dicDB.keys()
            total = self.size
        else:
            keys = random.sample(list(self.dicDB.keys()), random_max)
            total = len(keys)
        for i,k in enumerate(keys):
            print(str(i) + '/' + str(total))
            mol = self.dicDB[k][2]
            molfp = Chem.RDKFingerprint(mol.mol)
            molfp = np.asarray(list((molfp.ToBitString())))
            fps.append(molfp)
        return fps 

    def _get_total_mols(self):
        self.size = len(self.dicDB.keys())
        

class mol(object):
    """"""
    """"""
    def __init__(self ,smile = None, InChI = None, allparamaters = False):
        if smile != None and InChI == None:
            self.smile = smile
            self.mol = Chem.MolFromSmiles(self.smile)
        elif smile == None and InChI != None:
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
        elif smile != None and InChI != None:
            warnings.warn(f'Given that both SMILE and InChI format have been provided the molecule will be read with InChI')
            self.smile = smile
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
        else:
            raise ValueError('To initialize the molecule SMILE or InChI format is required')
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

if __name__ == '__main__':
    smile = 'Cc1cc(-c2csc(N=C(N)N)n2)cn1C'
    inchi = 'InChI=1S/C10H13N5S/c1-6-3-7(4-15(6)2)8-5-16-10(13-8)14-9(11)12/h3-5H,1-2H3,(H4,11,12,13,14)'
    #Testing init
    m = mol(smile=smile)
    m = mol(InChI=inchi)
    m = mol(smile=smile,InChI=inchi)
    #Testing BRICS decomposition
    m.get_BRICSdecomposition()
    m.get_clean_fragments()
    #Testing Tanimoto Similarity
    similarity = get_MolSimilarity(m,m)
    print(similarity)
