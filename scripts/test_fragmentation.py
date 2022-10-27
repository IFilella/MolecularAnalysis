from MolecularAnalysis import mollib

ligfile = '../tests/Mpro_complex_7rlsA1_super_ligand1.pdb'

lig = mollib.Mol(pdb=ligfile)

lig.get_BRICS_fragments(smiles=False)


