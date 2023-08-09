import os
import glob
from io import StringIO
import sys
from MolecularAnalysis import mol
from MolecularAnalysis import moldb
from MolecularAnalysis.fragments import fragmentation
from MolecularAnalysis.analysis import plot

class CapturingErr(list):
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stderr = self._stderr

class CapturingOut(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

if not os.path.exists('testout'): os.system('mkdir testout')

molobj=mol.Mol(smile='OC1=CC(C2=C(F)C3=NC(OC[C@]45C[C@@H](F)CN4CCC5)=NC(N6C[C@H]7N[C@H](CC7)C6)=C3C=N2)=C8C(C#C)=C(F)C=CC8=C1', name='mrtx1133')
if molobj.error==-1: raise ValueError('Error when loading SMILE')
molobj.saveToMol('testout/mrtx1133')

molobj=mol.Mol(InChI='InChI=1S/C33H31F3N6O2/c1-2-23-26(35)7-4-18-10-22(43)11-24(27(18)23)29-28(36)30-25(13-37-29)31(41-15-20-5-6-21(16-41)38-20)40-32(39-30)44-17-33-8-3-9-42(33)14-19(34)12-33/h1,4,7,10-11,13,19-21,38,43H,3,5-6,8-9,12,14-17H2/t19-,20-,21+,33+/m1/s1', name='mrtx1133')
if molobj.error==-1: raise ValueError('Error when loading InChi')

molobj=mol.Mol(pdb='../data/mrtx1133.pdb', name='mrtx1133')
if molobj.error==-1: raise ValueError('Error when loading pdb')
molobj.saveToPDB('testout/mrtx1133')

molobj=mol.Mol(rdkit=molobj.molrdkit, name='mrtx1133')
if molobj.error==-1: raise ValueError('Error when loading rdkit object')

molobj.getFingerPrint()
molobj.getParamaters()

scaffold=fragmentation.getScaffold(molobj)
scaffold=mol.Mol(rdkit=scaffold,name='mrtx1133_scaffold')
scaffold.saveToMol('testout/mrtx1133_scaffold')

fragments=fragmentation.get2DFragments(molobj)
fragments=fragmentation.get3DFragments(molobj)
fragments=fragmentation.get3DFragments(molobj,
                                       centers=[[1.460,4.876,-23.660],[5.129,7.368,-22.148],
                                                [6.885,2.564,-22.768],[-2.919,3.310,-23.881]],
                                       radius=[3,3,3,3],anchoring=False)
for i,frag in enumerate(molobj.fragments3D):
    _frag=mol.Mol(rdkit=frag)
    _frag.saveToPDB('testout/mrtx1133_frag%d'%i)
for i,box in enumerate(molobj.boxesFragments3D):
    for j,frag in enumerate(box):
        _frag=mol.Mol(rdkit=frag)
        _frag.saveToPDB('testout/mrtx1133_box%dfrag%d'%(i,j))

moldbobj=moldb.MolDB(smiDB='../data/moldb.smi',verbose=False)
moldbobj=moldb.MolDB(sdfDB='../data/moldb1.sdf',verbose=False)
moldbobj.saveToPickle('testout/moldb')
moldbobj.saveToSmi('testout/moldb.smi')
moldbobj=moldb.MolDB(molDB='testout/moldb.pickle',verbose=False)
pdblist=glob.glob('../data/pdblist/*')
moldbobj=moldb.MolDB(pdbList=pdblist,verbose=False)
for i,mol in enumerate(moldbobj.mols):
    mol.saveToMol('testout/mol%d'%i)
moldbobj=moldb.MolDB(molList=moldbobj.mols,verbose=False)

moldbobj.getParamatersDataFrame()
fps=moldbobj.getFingerprints()
fps=moldbobj.getFingerprints(random_max=8)

sim=moldb.getMolSimilarity(moldbobj.mols[0],moldbobj.mols[1])

moldbobj1=moldb.joinMolDBs([moldbobj,moldbobj],simt=1)
moldbobj2=moldb.joinMolDBs([moldbobj,moldbobj])

moldbobj3=moldb.intersectMolDBs(moldbobj1,moldbobj2,simt=1,verbose=False)

moldbobj3.getMatrixSimilarity()

fragmentation.getMolDB3DFragments(moldbobj3)
fragmentation.getMolDB3DFragments(moldbobj3,norm=False)

"""
moldbobj3.plotNPR('testout/NPR')
moldbobj3.plotNPR('testout/NPR_MW',zkey='MolWt')
moldbobj3.plotParamatersPCA('testout/paramatersPCA')
moldbobj3.plotRadar('testout/radar')

moldbobj3.plotPCA('testout/PCA')
with CapturingErr() as output:
    moldbobj3.plotPCA('testout/PCA2', k=2)
with CapturingOut() as output:
    moldbobj3.plotTSNE('testout/TSNE', perplexity=5)
with CapturingErr() as output:
    with CapturingOut() as output:
        moldbobj3.plotTSNE('testout/TSNE2', perplexity=5, k=2)
with CapturingErr() as output:
    moldbobj3.plotUMAP('testout/UMAP')
with CapturingErr() as output:
    moldbobj3.plotUMAP('testout/UMAP2',k=2)
with CapturingErr() as output:
    moldbobj3.plotTrimap('testout/Trimap', n_inliers=5)
with CapturingErr() as output:
    moldbobj3.plotTrimap('testout/Trimap2', n_inliers=5, k=2)
"""

moldbobj1=moldb.MolDB(sdfDB='../data/moldb1.sdf',verbose=False)
moldbobj2=moldb.MolDB(sdfDB='../data/moldb2.sdf',verbose=False)
moldbobj3=moldb.MolDB(sdfDB='../data/moldb3.sdf',verbose=False)

plot.plotPCA(dbs=[moldbobj1, moldbobj2, moldbobj3], names=['DB1','DB2','DB3'], output='testout/dbsPCA',
             colors=['r','b','g'], sizes=[10,5,5], alphas=[0.8,0.8,0.8], linewidths=0,
             markers=['o','X','.'], figsize=(6,6))
with CapturingOut() as output:
    plot.plotTSNE(dbs=[moldbobj1, moldbobj2, moldbobj3], names=['DB1','DB2','DB3'], output='testout/dbsTSNE',
                  colors=['r','b','g'], sizes=[10,5,5], alphas=[0.8,0.8,0.8], linewidths=0,
                  markers=['o','X','.'], figsize=(6,6), perplexity=8)
with CapturingErr() as error:
    with CapturingOut() as output:
        plot.plotUMAP(dbs=[moldbobj1, moldbobj2, moldbobj3], names=['DB1','DB2','DB3'], output='testout/dbsUMAP',
                      colors=['r','b','g'], sizes=[10,5,5], alphas=[0.8,0.8,0.8], linewidths=0,
                      markers=['o','X','.'], figsize=(6,6), n_neighbors=8)
with CapturingOut() as output:
    with CapturingErr() as error:
        plot.plotTrimap(dbs=[moldbobj1, moldbobj2, moldbobj3], names=['DB1','DB2','DB3'],
                        output='testout/dbsTrimap', colors=['r','b','g'], sizes=[10,5,5], alphas=[0.8,0.8,0.8],
                        linewidths=0, markers=['o','X','.'], figsize=(6,6))
