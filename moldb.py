from MolecularAnalysis.mol import Mol
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import dill as pickle
import os
import random
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import progressbar
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def joinMolDBs(dbs, simt=None):
    """
    Join a list of MolDB objects into a single MolDB object
    - dbs: list of MolDB objects
    - simt: similarity threshold to filter the molecules of the new
                 MolDB object (default None)
    """
    new_db = copy.copy(dbs[0])
    print('Joining %d moldbs' % len(dbs))
    for i, db in enumerate(dbs[1:]):
        print('%d/%d' % (i+2, len(dbs)))
        bar = progressbar.ProgressBar(maxval=len(db.dicDB.keys())).start()
        for j, key in enumerate(db.dicDB.keys()):
            bar.update(j)
            if key not in new_db.dicDB.keys():
                new_db.dicDB[key] = db.dicDB[key]
            else:
                old_eqSMILES = new_db.dicDB[key][0].split(',')
                new_eqSMILES = db.dicDB[key][0].split(',')
                total_eqSMILES = ','.join(list(set(old_eqSMILES + new_eqSMILES)))
                new_db.dicDB[key][0] = total_eqSMILES
                IDs = new_db.dicDB[key][1]
                new_db.dicDB[key][1] += ',%s' % IDs
        bar.finish()
    new_db._update()
    if simt:
        new_db.filterSimilarity(simt=simt, alg='Morgan4', verbose=False)
    return new_db

def intersectMolDBs(db1, db2, simt, alg='RDKIT', verbose=True):
    """
    Intersect two MolDB objects
    - db1: MolDB object
    - db2: MolDB object
    - simt: similarity threshold to filter the molecules of the new
            MolDB object
    - alg: Algrithm used to compute the Fingerprint (default Morgan4)
    - verbose: If True get additional details (default False)
    """
    db3 = copy.copy(db1)
    keepkeys_db1 = []
    keepkeys_db2 = []
    hitsSMILE = 0
    hitsSimilarity = 0
    if not db1.paramaters:
        print('getting paramaters db1')
        db1._getMolsParamaters()
        db1.paramaters = True
    if not db2.paramaters:
        db2._getMolsParamaters()
        print('getting paramaters db2')
        db2.paramaters = True
    bar = progressbar.ProgressBar(maxval=db1.size).start()
    for i, m1 in enumerate(db1.mols):
        bar.update(i)
        SMILE1 = m1.smile
        for j, m2 in enumerate(db2.mols):
            SMILE2 = m2.smile
            if SMILE1 == SMILE2:
                hitsSMILE += 1
                if verbose:
                    print(i, 'bySMILE', SMILE1, SMILE2)
                keepkeys_db1.append(SMILE1)
                keepkeys_db2.append(SMILE2)
                break
            if simt == 1:
                if m1.NumAtoms != m2.NumAtoms:
                    continue
                if m1.NOCount != m2.NOCount:
                    continue
                if m1.NHOHCount != m2.NHOHCount:
                    continue
                if m1.RingCount != m2.RingCount:
                    continue
                if m1.FractionCSP3 != m2.FractionCSP3:
                    continue
                if m1.NumAliphaticRings != m2.NumAliphaticRings:
                    continue
                if m1.NumAromaticRings != m2.NumAromaticRings:
                    continue
            similarity = getMolSimilarity(m1, m2, alg=alg)
            if similarity >= simt:
                hitsSimilarity += 1
                if verbose:
                    print(i, 'bySimilarity', SMILE1, SMILE2)
                keepkeys_db1.append(SMILE1)
                keepkeys_db2.append(SMILE2)
                break
    bar.finish()
    totalhits = hitsSMILE + hitsSimilarity
    sizedb1 = len(db1.dicDB.keys())
    sizedb2 = len(db2.dicDB.keys())
    if verbose:
        print('Total db1=%d, total db2=%d' % (sizedb1, sizedb2))
        print('Hits by SMILE: %d' % hitsSMILE)
        print('Hits by Similarity (threshold %.3f): %d'
            % (simt, hitsSimilarity))
        print('Total hits: %d' % totalhits)
        print('Percentage of elements of db1 in db2: %.3f'
            % (((float(totalhits)/float(sizedb1))*100)))
        print('Percentage of elements of db2 in db1: %.3f'
            % (((float(totalhits)/float(sizedb2))*100)))
    for i, k1 in enumerate(keepkeys_db1):
        k2 = keepkeys_db2[i]
        db3.dicDB[k1][0] += ',' + db2.dicDB[k2][0]
        db3.dicDB[k1][1] += ',' + db2.dicDB[k2][1]
    set_keepkeys_db3 = set(keepkeys_db1)
    set_keys_db3 = set(list(db3.dicDB.keys()))
    delkeys_db3 = list(set_keys_db3.difference(set_keepkeys_db3))
    for key in delkeys_db3:
        del db3.dicDB[key]
    db3._update()
    return db3

def getMolSimilarity(mol1, mol2, alg='Morgan4', nBits=2048, metric='Tanimoto'):
    """
    Get the molecular similarity between two molecules
    - mol1: Mol object
    - mol2: Mol object
    - alg: Algrithm used to compute the Fingerprint (default Morgan4)
    - nBits: Number of bits of the Fingerprint (default 2048)
    - metric: similarity metric (default Tanimoto)
    """
    fp1=mol1.getFingerPrint(alg=alg, nBits=nBits)
    fp2=mol2.getFingerPrint(alg=alg, nBits=nBits)
    if metric=='Tanimoto':
        return DataStructs.TanimotoSimilarity(fp1,fp2)
    elif  metric=='Dice':
        return DataStructs.DiceSimilarity(fp1,fp2)
    elif metric=='Cosine':
        return DataStructs.CosineSimilarity(fp1,fp2)
    elif metric=='Sokal':
        return DataStructs.SokalSimilarity(fp1,fp2)
    elif metric=='Russel':
        return DataStructs.RusselSimilarity(fp1,fp2)
    elif metric=='Kulczynski':
        return DataStructs.KulczynskiSimilarity(fp1,fp2)
    elif metric=='McConnaughey':
        return DataStructs.McConnaugheySimilarity(fp1,fp2)
    else:
        raise ValueError('Invalid Metric')

class MolDB(object):
    """""
    Class to store a database of molecules/fragments.
    It takes as input:
    - smiDB: to load the molecules from a SMILES file
    - sdfDB: to load the molecules from a sdf file
    - molDB: to load a precalculated MolDB object
    - pdbList: to load the molecules from a list of PDBs
    - molList: to load the molecules from a list of Mol objects
    - chirality: if False remove chirality (default True)
    - paramaters: if True calculate multiple molecular paramaters (default False)
    - verbose: If True get additional details (default False)
    The main attribute of the class is 'dicDB', a dicctionary with the molecules SMILES as keys and
    with lists of form [eqSMILES, IDs, Mol object] as values.
    """""
    def __init__(self, smiDB=None, molDB=None, sdfDB=None, pdbList=None,
                 molList=None, paramaters=False, chirality=True, verbose=False):
        self.paramaters = paramaters
        self.chirality = chirality
        # Load moldb object from a smiles file
        if smiDB != None and molDB == None and sdfDB == None and pdbList == None and molList == None:
            self.dicDB = {}
            db = open(smiDB, 'r')
            lines = db.readlines()
            count = 0
            counteq = 0

            bar = progressbar.ProgressBar(maxval=len(lines)).start()
            print('Loading MolDB from smi file')
            for i, line in enumerate(lines):
                bar.update(i)
                line = line.replace('\n', '')
                SMILE = line
                mol = Mol(smile=SMILE, allparamaters=self.paramaters,
                          chirality=self.chirality)
                if mol.error == -1:
                    continue
                if SMILE not in self.dicDB:
                    count += 1
                    self.dicDB[SMILE] = [SMILE, 'unk', mol]
                else:
                    counteq+=1
                    continue
                if verbose: print(count+1,SMILE)
            bar.finish()
            print('Repeated SMILES: %d'%counteq)
        # Load moldb object from a pickle containing a moldb object
        elif smiDB == None and molDB != None and sdfDB == None and pdbList == None and molList == None:
            print('Loading MolDB from moldb')
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
            with open(molDB, 'rb') as f:
                molDBobject = pickle.load(f)
            self.dicDB = molDBobject.dicDB
            self.paramaters = molDBobject.paramaters
            self.chirality = molDBobject.chirality
        # Load moldb object from an sdf file
        elif smiDB == None and molDB == None and sdfDB != None and pdbList == None and molList == None:
            self.dicDB = {}
            DB = Chem.SDMolSupplier(sdfDB, removeHs=False)
            counteq = 0
            print('Loading MolDB from sdf file')
            bar = progressbar.ProgressBar(maxval=len(DB)).start()
            for i, cpd in enumerate(DB):
                bar.update(i)
                try:
                    name = cpd.GetProp("_Name")
                except:
                    name = 'unk'
                if name == '' or name == 'unk':
                    try:
                        name = cpd.GetProp("Catalog ID")
                    except:
                        pass
                mol = Mol(rdkit=cpd, allparamaters=self.paramaters,
                          chirality=self.chirality, name=name)
                if mol.error==-1: continue
                SMILE=mol.smile
                eqSMILES=SMILE
                if SMILE not in self.dicDB:
                    self.dicDB[SMILE]=[eqSMILES,name,mol]
                else:
                    counteq+=1
                    self.dicDB[SMILE][1]+=',%s'%name
                    old_eqSMILES=self.dicDB[SMILE][0].split(',')
                    new_eqSMILES=eqSMILES.split(',')
                    total_eqSMILES=','.join(list(set(old_eqSMILES + new_eqSMILES)))
                    self.dicDB[SMILE][0]=total_eqSMILES
                if verbose: print(i+1,name,SMILE)
            bar.finish()
            print('Repeated SMILES: %d'%counteq)
        # Load moldb obeject from a list of pdbs
        elif smiDB==None and molDB==None and sdfDB==None and pdbList!=None and molList==None:
            self.dicDB={}
            counteq=0
            print('Loading MolDB from pdb list')
            bar = progressbar.ProgressBar(maxval=len(pdbList)).start()
            for i,pdb in enumerate(pdbList):
                bar.update(i)
                pdb_name = os.path.basename(pdb)
                mol = Mol(pdb=pdb, allparamaters=self.paramaters, chirality=self.chirality, name=pdb_name)
                if mol.error==-1: continue
                SMILE=mol.smile
                eqSMILES = SMILE
                IDs=os.path.basename(pdb).replace(".pdb","")
                if SMILE not in self.dicDB:
                    self.dicDB[SMILE]=[eqSMILES,IDs,mol]
                else:
                    counteq+=1
                    self.dicDB[SMILE][1]+=',%s'%IDs
                    old_eqSMILES=self.dicDB[SMILE][0].split(',')
                    new_eqSMILES=eqSMILES.split(',')
                    total_eqSMILES=','.join(list(set(old_eqSMILES + new_eqSMILES)))
                    self.dicDB[SMILE][0]=total_eqSMILES
                if verbose: print(i+1,IDs,SMILE)
            bar.finish()
            print('Unique molecules %d.\nRepeated SMILES: %d'
                % (len(self.dicDB.keys()), counteq))
        # Load moldb object from a list of mol objects
        elif smiDB==None and molDB==None and sdfDB==None and pdbList==None and molList!=None:
            self.dicDB = {}
            counteq = 0
            print('Loading MolDB from a list of Mol objects')
            bar = progressbar.ProgressBar(maxval=len(molList)).start()
            for i, molobj in enumerate(molList):
                bar.update(i)
                smile = molobj.smile
                name = molobj.name
                if smile not in self.dicDB:
                    self.dicDB[smile] = [smile, name, molobj]
                else:
                    counteq += 1
                    self.dicDB[smile][1] += ',%s' % name
                    old_eqSMILES = self.dicDB[smile][0].split(',')
                    new_eqSMILES = [smile]
                    total_eqSMILES = ','.join(list(set(old_eqSMILES + new_eqSMILES)))
                    self.dicDB[smile][0] = total_eqSMILES
                if verbose:
                    print(i+1, name, smile)
            bar.finish()
            print('Unique molecules %d.\nRepeated SMILES: %d'
                % (len(self.dicDB.keys()), counteq))
        else:
            raise KeyError('Provide only a smiDB, a molDB object,\
            a sdfDB, a pdbList or a molList')
        self._update()

    def _update(self):
        """
        Compute and update core atributtes of the object
        """
        self.smiles=self.dicDB.keys()
        mols=[]
        eqsmiles=[]
        IDs=[]
        for smile in self.smiles:
            mols.append(self.dicDB[smile][2])
            IDs.append(self.dicDB[smile][1])
            eqsmiles.append(self.dicDB[smile][0])
        self.mols=mols
        self.eqsmiles=eqsmiles
        self.IDs=IDs
        self._getTotalMols()
        self.table=None

    def saveToPickle(self,output):
        """
        Save the molDB object with pickle
        - output: output file name (without format)
        """
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with open(output+'.pickle', 'wb') as handle:
            pickle.dump(self, handle)

    def saveToSmi(self,output):
        """
        Save the molDB object to smi file
        - output: output file name (without format)
        """
        f=open(output,'w')
        for smile in self.smiles:
            f.write('%s\n'%smile)
        f.close()

    def saveToSdf(self,output):
        """
        Save the molDB object to sdf file
        - output: output file name (without format)
        """
        with Chem.SDWriter(output + '.sdf') as w:
            for k in self.dicDB.keys():
                molrdkit=self.dicDB[k][2].molrdkit
                ID=self.dicDB[k][1]
                molrdkit.SetProp("_Name",ID)
                w.write(molrdkit,)

    def getFingerprints(self, alg='Morgan4', nBits=2048, random_max=None, verbose=False):
        """
        Get molecular FingerPrint for all molecules
        - alg: Algorithm used to compute the Fingerprint (default Morgan4)
        - nBits: Number of bits of the Fingerprint (default 2048)
        - random_max: If an integer is given the fingerprint is going to be
                      computed only for this number of molecules (default None)
        - verbose: If True get additional details (default False)
        """
        fps=[]
        if random_max==None:
            keys=self.dicDB.keys()
            total=self.size
        elif random_max > self.size:
            keys=self.dicDB.keys()
            total=self.size
        else:
            keys=random.sample(list(self.dicDB.keys()), random_max)
            total=len(keys)
        for i,k in enumerate(self.dicDB.keys()):
            if verbose: print(alg + ': ' + str(i) + '/' + str(total))
            if k in keys:
                mol=self.dicDB[k][2]
                molfp=mol.getFingerPrint(alg,nBits)
                molfp=np.asarray(list((molfp.ToBitString())))
            else:
                molfp=None
            fps.append(molfp)
        self.fingerprints=fps
        return fps

    def get_molecular_paramater_list(self, paramater):
        if not hasattr(self.mols[0], paramater):
            raise ValueError('%s is not a Mol attribute' % paramater)
        paramater_list = []

        for mol in self.mols:
            value = getattr(mol, paramater)
            paramater_list.append(value)

        setattr(self, paramater, paramater_list)

        return paramater_list

    def getParamatersDataFrame(self):
        """
        Construct a DataFrame whith several molecular\
        paramaters for each molecule
        """
        if self.paramaters:
            pass
        else:
            self._getMolsParamaters()
            self.paramaters = True
        df = pd.DataFrame()
        paramaters = ['MolWt', 'LogP', 'NumHAcceptors', 'NumHDonors',
                      'NumHeteroatoms', 'NumRotatableBonds', 'NumHeavyAtoms',
                      'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                      'NumAliphaticRings', 'NumAromaticCarbocycles',
                      'NumAromaticHeterocycles', 'NumAromaticRings',
                      'RingCount', 'FractionCSP3', 'TPSA', 'NumAtoms',
                      'NPR1', 'NPR2', 'InertialShapeFactor',
                      'RadiusOfGyration']
        df['smile'] = self.smiles
        df['id'] = self.IDs
        for paramater in paramaters:
            paramater_list = self.get_molecular_paramater_list(paramater)
            df[paramater] = paramater_list

        self.df = df

    def filterSimilarity(self,simt=1, alg='Morgan4',verbose=True):
        """
        Filter the molcules of the MolDB object by a similarity treshold
        - simt: similarity threshold
        - alg: Algorithm used to compute the Fingerprint (default Morgan4)
        - verbose: If True get additional details (default False)
        """
        if not self.paramaters:
            self._getMolsParamaters()
            self.paramaters=True
        new_dicDB={}
        count=0
        for i,key1 in enumerate(self.dicDB.keys()):
            count+=1
            SMILE1=key1
            eqSMILES1=self.dicDB[key1][0]
            IDs1=self.dicDB[key1][1]
            mol1=self.dicDB[key1][2]
            if len(new_dicDB.keys())==0:
                new_dicDB[SMILE1]=[eqSMILES1,IDs1,mol1]
            for j,key2 in enumerate(new_dicDB.keys()):
                SMILE2=key2
                eqSMILES2=new_dicDB[key2][0]
                IDs2=new_dicDB[key2][1]
                mol2=new_dicDB[key2][2]
                if mol1.NumAtoms!=mol2.NumAtoms or mol1.NOCount!=mol2.NOCount or mol1.RingCount!=mol2.RingCount or mol1.NHOHCount!=mol2.NHOHCount or mol1.FractionCSP3!=mol2.FractionCSP3 or mol1.NumAliphaticRings!=mol2.NumAliphaticRings or  mol1.NumAromaticRings!=mol2.NumAromaticRings:
                    if j==len(new_dicDB.keys())-1:
                        new_dicDB[SMILE1]=[eqSMILES1,IDs1,mol1]
                        break
                    else:
                        continue
                else:
                    similarity=getMolSimilarity(mol1,mol2,alg=alg)
                    if j==len(new_dicDB.keys())-1 and similarity >= simt:
                        if verbose: print("0:" + new_dicDB[SMILE2][0] + " " + SMILE1)
                        _eqSMILES1=eqSMILES1.split(',')
                        _eqSMILES2=eqSMILES2.split(',')
                        new_eqSMILES=','.join(list(set(_eqSMILES1 + _eqSMILES2)))
                        new_dicDB[SMILE2][0]=new_eqSMILES
                        _IDs1=IDs1.split(',')
                        _IDs2=IDs2.split(',')
                        new_IDs=','.join(list(set(_IDs1 + _IDs2)))
                        new_dicDB[SMILE2][1]=new_IDs
                    elif similarity >= simt:
                        if verbose: print("1:" + new_dicDB[SMILE2][0] + " " + SMILE1)
                        _eqSMILES1=eqSMILES1.split(',')
                        _eqSMILES2=eqSMILES2.split(',')
                        new_eqSMILES=','.join(list(set(_eqSMILES1 + _eqSMILES2)))
                        new_dicDB[SMILE2][0]=new_eqSMILES
                        _IDs1=IDs1.split(',')
                        _IDs2=IDs2.split(',')
                        new_IDs=','.join(list(set(_IDs1 + _IDs2)))
                        new_dicDB[SMILE2][1]=new_IDs
                        break
                    else:
                        continue
            if verbose:
                print('Unique molecules after filtering: %d'%len(new_dicDB.keys()))
                print('Analized molecules: %d'%count)
            else:
                if count % 500==0:  print('Analized compounds: %d'%count)
        self.dicDB=new_dicDB
        self._update()

    def _get_similarity_matrix(self,
                               radius=4,
                               metric='Tanimoto',
                               verbose=False):
        """

        """
        # Get Morgan Fingerprints
        fps_vecs = []
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
        print('Computing structural fingerprint')
        bar = progressbar.ProgressBar(maxval=len(self.mols)).start()
        for i, mol in enumerate(self.mols):
            bar.update(i)
            fps_vecs.append(fpgen.GetSparseCountFingerprint(mol.molrdkit))
        bar.finish()

        # Get structural similarity matrix
        similarity_matrix = []
        print('Computing similarity matrix')
        bar = progressbar.ProgressBar(maxval=len(fps_vecs)).start()
        for i, fp in enumerate(fps_vecs):
            bar.update(i)
            similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(fp,
                                                                        fps_vecs))
        bar.finish()

        names = []
        for mol in self.mols:
            name = mol.molrdkit.GetProp('_Name')
            names.append(name)

        self.simmatrix = pd.DataFrame(similarity_matrix,
                                      index=names,
                                      columns=names)

    def plot_similarity_clustermap(self, outname):
        """

        """
        if not hasattr(self, 'simmatrix'):
            self._get_similarity_matrix()

        plt.subplots(dpi=300, layout='constrained')
        g = sns.clustermap(1-self.simmatrix, cmap='OrRd')
        ax = g.ax_heatmap
        ax.set_xlabel('Molecule')
        ax.set_ylabel('Molecule')
        plt.savefig(outname)

    def _get_dendrogram(self):
        """

        """
        if not hasattr(self, 'simmatrix'):
            self._get_similarity_matrix()

        self.simdendrogram = linkage(self.simmatrix, 'average', metric='euclidean')

    def get_dendrogram_clusters(self, max_d):
        """

        """
        if not hasattr(self, 'simdendrogram'):
            self._get_dendrogram()

        clusters = fcluster(self.simdendrogram, max_d, criterion='distance')
        self.simdendrogram_clusters = clusters

        ids = self.simmatrix.columns.tolist()

        clusters_dic = {}
        for i, id in enumerate(ids):
            cluster = self.simdendrogram_clusters[i]
            if cluster not in clusters_dic:
                clusters_dic[cluster] = [id]
            else:
                clusters_dic[cluster].append(id)

        self.simdendrogram_clusters_dic = clusters_dic

        return self.simdendrogram_clusters_dic

    def plot_dendrogram(self, outname, max_d, annotate_above, p=None):
        """

        """
        if not hasattr(self, 'simdendrogram'):
            self._get_dendrogram()

        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical clustering dendrogram')
        plt.xlabel('sample index', fontsize=16)
        plt.ylabel('distance', fontsize=16)

        if p:
            self._plot_fancy_dendrogram(self.simdendrogram,
                                        leaf_rotation=90.,
                                        leaf_font_size=8.,
                                        labels=self.IDs,
                                        max_d=max_d,
                                        annotate_above=annotate_above,
                                        p=p,
                                        truncate_mode='lastp',
                                        show_contracted=True)
        else:
            self._plot_fancy_dendrogram(self.simdendrogram,
                                        leaf_rotation=90.,
                                        leaf_font_size=8.,
                                        labels=self.IDs,
                                        max_d=max_d,
                                        annotate_above=annotate_above)

        ax = plt.gca()
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='major', labelsize=14)
        plt.savefig(outname, dpi=300)

    def _plot_fancy_dendrogram(self, *args, **kwargs):
        """

        """
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'],
                               ddata['dcoord'],
                               ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    def getMatrixSimilarity(self, alg='Morgan4', metric='Tanimoto',verbose=False):
        """
        Get pairwise similarity matrix of all molecules in the MolDB obj
        - alg: Algrithm used to compute the Fingerprint (default Morgan4)
        - metric: similarity metric (default Tanimoto)
        - verbose: If True get additional details (default False)
        """
        simmatrix = np.zeros((self.size, self.size))
        bar = progressbar.ProgressBar(maxval=len(self.mols)).start()
        for i, mol in enumerate(self.mols):
            bar.update(i)
            for j, mol in enumerate(self.mols):
                if i < j:
                    if verbose:
                        print('%d/%d' % (i+1, self.size))
                    sim = getMolSimilarity(self.mols[i],
                                           self.mols[j],
                                           alg=alg,
                                           metric=metric)
                    simmatrix[i][j] = sim
                    simmatrix[j][i] = sim
                elif i == j:
                    simmatrix[i][i] = 1
                else:
                    continue
        bar.finish()
        self.simmatrix = simmatrix

    def plotNPR(self,output, zkey=None):
        """
        Generate an NPR plot to investigate the shapelike form of the molecules in the MolDB
        - output: plot output name
        - zkey: extra Z dimension to select among the df columns
        """
        if isinstance(self.df, pd.DataFrame):
            pass
        else:
            self.getParamatersDataFrame()

        plt.rcParams['axes.linewidth']=1.5
        plt.figure()

        if zkey==None:
            ax=sns.scatterplot(data=self.df, x='NPR1',y='NPR2',s=25,linewidth=0.5,alpha=1)
        else:
            x=self.df['NPR1'].tolist()
            y=self.df['NPR2'].tolist()
            z=self.df[zkey].tolist()
            ax= plt.scatter(x=x,y=y,c=z,data=self.df,s=8,linewidth=0.5,alpha=0.75)
            plt.colorbar()

        x1, y1=[0.5, 0], [0.5, 1]
        x2, y2=[0.5, 1], [0.5, 1]
        x3, y3=[0,1],[1,1]

        plt.plot(x1, y1,x2,y2,x3,y3,c='gray',ls='--',lw=1)
        plt.xlabel ('NPR1',fontsize=15,fontweight='bold')

        plt.ylabel ('NPR2',fontsize=15,fontweight='bold')

        if zkey==None:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            pass

        plt.text(0, 1.01,s='Rod',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')
        plt.text(1, 1.01,s='Sphere',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')
        plt.text(0.5, 0.49,s='Disc',fontsize=16,horizontalalignment='center',verticalalignment='center',fontweight='bold')

        plt.tick_params ('both',width=2,labelsize=14)
        plt.tight_layout()
        plt.savefig(output+'.png',dpi=300)

    def plotParamatersPCA(self, output, verbose=False):
        """
        Plot an sphere wiht the 2 principal components of a PCA done with the df paramaters
        - output: plot output name
        - verbose: If True get additional details (default False)
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        if isinstance(self.df, pd.DataFrame):
            pass
        else:
            self.getParamatersDataFrame()

        descriptors=self.df[['MolWt', 'LogP','NumHeteroatoms','RingCount','FractionCSP3', 'TPSA','RadiusOfGyration']].values
        descriptors_std=StandardScaler().fit_transform(descriptors)
        pca=PCA()
        descriptors_2d=pca.fit_transform(descriptors_std)
        descriptors_pca= pd.DataFrame(descriptors_2d)
        descriptors_pca.index=self.df.index
        descriptors_pca.columns=['PC{}'.format(i+1) for i in descriptors_pca.columns]
        descriptors_pca.head(5)

        if verbose:
            print(pca.explained_variance_ratio_) #Let's plot PC1 vs PC2
            print(sum(pca.explained_variance_ratio_))

        scale1=1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
        scale2=1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))

        descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
        descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

        plt.rcParams['axes.linewidth']=1.5
        plt.figure(figsize=(6,6))

        ax=sns.scatterplot(data=descriptors_pca, x='PC1_normalized',y='PC2_normalized',s=20,palette=sns.color_palette("Set2", 3),linewidth=0.2,alpha=1)

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
        if verbose:
            print(lab)
            print(l)
            print(np.linalg.norm(l,axis=1))

        n=l.shape[0]
        for i in range(n):
            plt.arrow(0, 0, l[i,0], l[i,1],color= 'k',alpha=0.5,linewidth=1.8,head_width=0.025)
            plt.text(l[i,0]*1.25, l[i,1]*1.25, lab[i], color='k',va='center', ha='center',fontsize=16)

        circle=plt.Circle((0,0), 1, color='gray', fill=False,clip_on=True,linewidth=1.5,linestyle='--')
        plt.tick_params ('both',width=2,labelsize=18)

        ax.add_artist(circle)
        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.tight_layout()

        plt.savefig(output+'.png',dpi=300)

    def plotRadar(self,output=None):
        """
        Plot a biovailability radar plot
        - output: plot output name
        """
        if hasattr(self, 'df'):
            pass
        else:
            self.getParamatersDataFrame()

        data=pd.DataFrame()

        data['MolWt']=[i/500 for i in self.df['MolWt']]
        data['LogP']=[i/5 for i in self.df['LogP']]
        data['nHA']=[i/10 for i in self.df['NumHAcceptors']]
        data['nHD']=[i/5 for i in self.df['NumHDonors']]
        data['nRotB']=[i/10 for i in self.df['NumRotatableBonds']]
        data['TPSA']=[i/140 for i in self.df['TPSA']]

        categories=list(data.columns)
        N=len(categories)
        values=data[categories].values[0]
        values=np.append(values,values[:1])
        angles=[n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        Ro5_up=[1,1,1,1,1,1,1] #The upper limit for bRo5
        Ro5_low=[0.5,0.1,0,0,0,0.25,0.5]  #The lower limit for bRo5

        fig=plt.figure()

        fig, ax=plt.subplots(subplot_kw={'projection': 'polar'})

        plt.xticks(angles[:-1], categories,color='k',size=15,ha='center',va='top',fontweight='book')

        plt.tick_params(axis='y',width=4,labelsize=6, grid_alpha=0.05)

        ax.set_rlabel_position(0)

        all_values=[]
        for i in data.index:
            values=data[categories].values[i]
            values=np.append(values,values[:1])
            all_values.append(values)
            plt.plot(angles, values, linewidth=0.6 ,color='steelblue',alpha=0.5)

        all_values=np.asarray(all_values)
        average_values =np.mean(all_values,axis=0)
        plt.plot(angles, average_values, linewidth=1 ,linestyle='-',color='orange')

        ax.grid(axis='y',linewidth=1.5,linestyle='dotted',alpha=0.8)
        ax.grid(axis='x',linewidth=2,linestyle='-',alpha=1)

        plt.plot(angles, Ro5_up, linewidth=2, linestyle='-',color='red')
        plt.plot(angles, Ro5_low, linewidth=2, linestyle='-',color='red')

        plt.tight_layout()
        plt.savefig(output,dpi=300)

    def plotPCA(self, output, random_max=None, alg='Morgan4', k=None):
        """
        Performe a PCA with the fingerprint of the molecules of MolDB an plot
        the two first components
        - output: plot name
        - random_max: If an integer is given the PCA is going to be done
                      only for this number of molecules (default None)
        - alg: Algorithm used to compute the Fingerprint (default Morgan4)
        - k: clusterize the PCA results using k-clusters obtained
                  with kmeans algorithm
        """
        from sklearn.decomposition import PCA
        fps=self.getFingerprints(alg, random_max)
        fps=[fp for fp in fps if fp is not None]
        pca=PCA(n_components=2)
        pca_results=pca.fit_transform(fps)
        self._plotReducer(pca_results, output, k)

    def plotTSNE(self, output, random_max=None, n_iter=1500, perplexity=30,
                 alg='Morgan4', k=None, early_exaggeration=12):
        """
        Perform a tSNE with the fingerprint of the molecules of MolDB an plot the
        two first components
        - output: plot name
        - random_max: If an integer is given the TSNE is going to be done
                    only for this number of molecules (default None)
        - alg: Algorithm used to compute the Fingerprint (default Morgan4)
        - k: clusterize the TSNE results using k-clusters obtained
                with kmeans algorithm
        - n_iter: Maximum number of iterations (default 1500)
        - perplexity: paramater related to the number of nearest neighbors (default 30)
        - early_exaggeration: paramater that controls how tight natural clusters
                              in the original space are in the embedded space and how much space
                              will be between them (default 12)
        """
        from sklearn.manifold import TSNE
        fps=self.getFingerprints(alg, random_max)
        fps=[fp for fp in fps if fp is not None]
        tsne=TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                  n_iter=n_iter, perplexity=perplexity,metric='hamming',
                  early_exaggeration=early_exaggeration)
        tsne_results=tsne.fit_transform(np.asarray(fps))
        self._plotReducer(tsne_results,output,k)

    def plotUMAP(self, output, random_max=None, alg='Morgan4', k=None):
        """
        Perform a UMAP with the fingerprint of the molecules of MolDB an plot the results
        - output: plot name
         - random_max: If an integer is given the UMAP is going to be done
                     only for this number of molecules (default None)
         - alg: Algorithm used to compute the Fingerprint (default Morgan4)
         - k: clusterize the UMAP results using k-clusters obtained
                 with kmeans algorithm
        """
        import umap as mp
        fps=self.getFingerprints(alg, random_max)
        fps=[fp for fp in fps if fp is not None]
        umap=mp.UMAP(n_neighbors=50, n_epochs=5000, min_dist= 0.5, metric='hamming')
        UMAP_results=umap.fit_transform(fps)
        self._plotReducer(UMAP_results,output,k)

    def plotTrimap(self, output, random_max=None, alg='Morgan4', k=None, n_inliers=30):
        """
        Perform a tSNE with the fingerprint of the molecules of MolDB an plot the
        two first components
        - output: plot name
        - random_max: If an integer is given the Trimap is going to be done
                    only for this number of molecules (default None)
        - alg: Algorithm used to compute the Fingerprint (default Morgan4)
        - k: clusterize the Trimap results using k-clusters obtained
                with kmeans algorithm
        - n_inliers: (default 30)
        """
        import trimap
        fps=self.getFingerprints(alg, random_max)
        fps=[fp for fp in fps if fp is not None]
        tri=trimap.TRIMAP(n_dims=2, distance='hamming', n_iters=2500, n_inliers=n_inliers,
                          n_outliers=5, weight_temp=0.6)
        tri_results=tri.fit_transform(np.asarray(fps))
        self._plotReducer(tri_results,output,k)

    def _plotReducer(self, reducer_results, output, k=None):
        """
        Plot the dimensional reductions results (PCA, UMAP, ...)
        - reducer_results: dimensional reduction data
        - output: plot name
        - k: clusterize the PCA results using k-clusters obtained
                  with kmeans algorithm
        """
        plt.figure()
        if k!=None:
            labels,centroids,clusters=self._doKmeans(k,reducer_results)
            df=pd.DataFrame(dict(xaxis=reducer_results[:,0],
                                 yaxis=reducer_results[:,1], cluster=labels))
            sns.scatterplot(data=df, x='xaxis', y='yaxis', hue='cluster', alpha=0.8,
                            s=15,style='cluster',palette=sns.color_palette("hls", k))
            plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
        else:
            df=pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1]))
            sns.scatterplot(data=df, x='xaxis', y='yaxis', alpha=0.8, s=15)
        plt.savefig(output+'.png',dpi=300)

    def _doKmeans(self, k, data):
        """
        Clusterize MolDB molecules using the kmeans algorithm
        - k: number of clusters
        - data:
        """
        from sklearn.cluster import KMeans
        model=KMeans(n_clusters=k, n_init=1,init="k-means++")
        labels=model.fit_predict(data)
        centroids=model.cluster_centers_
        clusters=[]
        for i in range(k):
            indexes=np.where(np.array(labels)==i)[0]
            clusters.append(indexes)
        #clusters=np.asarray(clusters)
        return labels, centroids, clusters

    def _getTotalMols(self):
        """
        Get total number of molecules in the molDB object
        """
        self.size=len(self.dicDB.keys())

    def _getMolsParamaters(self):
        """
        Get several molecular paramaters using rdkit functions for each molecule
        """
        if self.paramaters: return
        for k in self.dicDB.keys():
            mol=self.dicDB[k][2]
            mol.getParamaters()
        self.paramaters = True

