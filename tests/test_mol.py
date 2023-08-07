import os
from MolecularAnalysis import mol
from MolecularAnalysis.fragments import fragmentation

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
                                       radius=[3,3,3,3],verbose=True,anchoring=False)
for i,frag in enumerate(molobj.fragments3D):
    _frag=mol.Mol(rdkit=frag)
    _frag.saveToPDB('testout/mrtx1133_frag%d'%i)
for i,box in enumerate(molobj.boxesFragments3D):
    for j,frag in enumerate(box):
        _frag=mol.Mol(rdkit=frag)
        _frag.saveToPDB('testout/mrtx1133_box%dfrag%d'%(i,j))
