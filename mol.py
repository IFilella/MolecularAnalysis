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
from umap import UMAP
import time
import trimap

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
        for i,k1 in enumerate(keepkeys_db1):
            k2 = keepkeys_db2[i]
            db3.dicDB[k1][0] += ',' + db2.dicDB[k2][0]
            db3.dicDB[k1][1] += ',' + db2.dicDB[k2][1]
        set_keepkeys_db3 = set(keepkeys_db1)
        set_keys_db3 = set(list(db3.dicDB.keys()))
        delkeys_db3 = list(set_keys_db3.difference(set_keepkeys_db3))
        for key in delkeys_db3:
            del db3.dicDB[key]
        db3.print_molDB(output)
        db3.save_molDB(output)

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

def _prepare_reducer_data(dbs,names,random_max, delimiter, fpsalg):
    X = []
    Y = []
    for i,db in enumerate(dbs):
        if delimiter == None:
            name = names[i]
        else:
            #name = names[i].split(delimiter)[0]
            name = '_'.join(names[i].split(delimiter)[0:2])
        print(name)
        fps = db.get_fingerprints(fpsalg, random_max)
        X.extend(fps)
        Y.extend([name]*len(fps))
    X = np.asarray(X)
    return X, Y

def _plot_reducer_data(reducer_results,Y,output):
    df = pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1],  molDB = Y))
    plt.figure()
    sns.scatterplot('xaxis', 'yaxis', data=df, hue='molDB',alpha = 0.5, s=3,style='molDB')
    if output != None:
        plt.savefig(output+'.png',dpi=300)
    plt.show()

def plot_trimap(dbs,names,output=None, random_max = 5000, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer_data(dbs,names,random_max, delimiter, fpsalg)
    print('Computing trimap')
    embedding = trimap.TRIMAP()
    trimap_results = embedding.fit_transform(X)
    print('Shape of trimap_results: ', trimap_results.shape)
    _plot_reducer_data(reducer_results = trimap_results, Y=Y, output=output)

def plot_UMAP(dbs,names,output=None, random_max = 5000, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer_data(dbs,names,random_max, delimiter, fpsalg)
    print('Computing UMAP')
    reducer = UMAP(n_neighbors=100, n_epochs=1000)
    UMAP_results = reducer.fit_transform(X)
    print('Shape of UMAP_results: ', UMAP_results.shape)
    _plot_reducer_data(reducer_results = UMAP_results, Y=Y, output=output)

def plot_TSNE(dbs,names,output=None, random_max = 5000, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer_data(dbs,names,random_max, delimiter, fpsalg)
    print('Computing TSNE')
    tsne_results = TSNE(n_components=2, verbose = 1, learning_rate='auto',init='pca').fit_transform(X)
    _plot_reducer_data(reducer_results = tsne_results, Y=Y, output=output)

def plot_PCA(dbs,names,output=None, random_max = 5000, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer_data(dbs,names,random_max, delimiter, fpsalg)
    print('Computing PCA')
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    _plot_reducer_data(reducer_results = pca_results, Y=Y, output=output)

def read_compoundDB(data):
    compoundDB = Chem.SDMolSupplier(data)
    return compoundDB

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
            auxmol = mol(smile=SMILE)
            auxerror = auxmol._remove_anchorings()
            
            #Check eqSMILES and define new_eqSMILES
            new_eqSMILES = []
            for eqSMILE in eqSMILES:
                auxmol2 = mol(smile=eqSMILE)
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

    def get_FingerPrint(self,alg='RDKIT'):
        if alg == 'RDKIT':
            self.FingerPrint = Chem.RDKFingerprint(self.mol)
        elif alg == 'Morgan2':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 1, nBits=2048)
        elif alg == 'Morgan4':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
        elif alg == 'Morgan8':
            self.FingerPrint = AllChem.GetMorganFingerprintAsBitVect(self.mol, 4, nBits=2048)
        else:
            raise ValueError('Invalid fingerprint algorithm')
        return self.FingerPrint


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
