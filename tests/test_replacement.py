from MolecularAnalysis import mol,moldb
from MolecularAnalysis.fragments import replacement
from frag_hop.main import run_replacement
from prody import *
import os
import glob

complex_pdb = '../data/7RPZ_mrtx1133.pdb'
fragsS2 = moldb.MolDB(sdfDB='../data/S2frags.sdf', verbose=False)
fragsS3 = moldb.MolDB(sdfDB='../data/S3frags.sdf', verbose=False)
fragsS4 = moldb.MolDB(sdfDB='../data/S4frags.sdf', verbose=False)

if not os.path.exists('testout/replacement'): os.system('mkdir testout/replacement')

for i,frag in enumerate(fragsS2.mols):
    frag.molrdkit.SetProp('_Name','S2_ligfrag_%d'%i)
    replacement.replaceFrag(frag, complex_pdb = complex_pdb,
                            output = 'testout/replacement/S2_ligfrag_%d'%i,
                            anchoring_lig = 'O1-C13', chain_id = 'L', verbose = False)
for i, frag in enumerate(fragsS3.mols):
    frag.molrdkit.SetProp('_Name','S3_ligfrag_%d'%i)
    replacement.replaceFrag(frag, complex_pdb = complex_pdb,
                            output = 'testout/replacement/S3_ligfrag_%d'%i,
                            anchoring_lig = 'N4-C14', chain_id = 'L', verbose = False)
for i, frag in enumerate(fragsS4.mols):
    frag.molrdkit.SetProp('_Name','S4_ligfrag_%d'%i)
    replacement.replaceFrag(frag, complex_pdb = complex_pdb,
                            output = 'testout/replacement/S4_ligfrag_%d'%i,
                            anchoring_lig = 'C18-C16', chain_id = 'L', verbose = False)
