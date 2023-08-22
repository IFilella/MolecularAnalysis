from MolecularAnalysis import mol, moldb
from frag_hop.data.parameters import atom_constants
from frag_hop.main import run_replacement
from prody import *
from rdkit import Chem
import os

def replaceFrag(frag, complex_pdb, anchoring_lig, chain_id,
                output='ligfrag', verbose = False):
    """
    Replace a fragment of a ligand for another fragment
    - frag: fragment to add as a Mol object with dummy atoms ('*' or 'Du') as anchoring points
    - complex_pdb: target + ligand complex pdb
    - anchoring_lig: ligand anchoring bond from the outside to the inside 'AA-AA' e.g 'C1-O1'
    - chain_id: chain ID of the ligand
    """
    # frag anchoring bond from the outside (dummy) to the inside
    anchorings_frag = prepareFrag(frag, verbose = verbose)
    frag.saveToPDB('tmp')
    pdb_frag = parsePDB('tmp.pdb')
    atom_names_frag = pdb_frag.getNames()
    for i,anchoring_frag in enumerate(anchorings_frag):
        anchoring_frag = list(atom_names_frag[list(anchoring_frag)])
        anchoring_frag = '%s-%s'%(anchoring_frag[1], anchoring_frag[0])
        if verbose:
            print(complex_pdb)
            print(anchoring_lig, anchoring_frag)
        run_replacement(complex_pdb = complex_pdb, fragment_pdb='tmp.pdb',
                        connectivity1 = anchoring_lig, output = '%s_%d'%(output,i),
                        connectivity2 = anchoring_frag, chain_id = chain_id)
        _cleanReplaceFrag(output + '_' + str(i))
        # Join initial target structure with the new ligand
        complexPDB = parsePDB(complex_pdb)
        target = complexPDB.select('protein').copy()
        newligandPDB = parsePDB('%s_%d/LIG.pdb'%(output, i))
        newligand = newligandPDB.select('hetero').copy()
        newcomplex = newligand + target
        writePDB('%s_%d/COMPLEX.pdb'%(output, i), newcomplex)
    os.system('rm tmp.pdb')

def prepareFrag(frag, verbose = False):
    """
    Prepare a fragment before adding it into a molecule. Dummy atoms such as '*' or 'Du'
    are replaced by Hydrogen atoms and its bond distance is correspondingly corrected.
    A list of tuples of (dummyatom_idx, atom_connected_to_dummyatom_idx) is returned
    - frag: fragment as a Mol object
    """
    idxs = list()
    for atom in frag.molrdkit.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol == '*' or atom_symbol == 'Du':
            atom_idx = atom.GetIdx()
            atom_bonds = atom.GetBonds()
            if verbose: print(atom_idx)
            if len(atom_bonds) != 1:
                raise ValueError('Incorrect number of bonds (!=1) in the dummy atom')
            bond = atom_bonds[0]
            bond_type = bond.GetBondType()
            begin_atom, begin_symbol, begin_idx = \
                bond.GetBeginAtom(), bond.GetBeginAtom().GetSymbol(), bond.GetBeginAtom().GetIdx()
            end_atom, end_symbol, end_idx = \
                bond.GetEndAtom(), bond.GetEndAtom().GetSymbol(), bond.GetEndAtom().GetIdx()
            if verbose:
                print(begin_symbol,begin_idx)
                print(end_symbol,end_idx)
            if end_symbol == '*' or end_symbol == 'Du':
                new_distance = atom_constants.BONDING_DISTANCES['H',\
                                                                begin_symbol,str(bond_type).lower()]
                Chem.rdMolTransforms.SetBondLength(frag.molrdkit.GetConformer(),
                                                   end_idx, begin_idx, new_distance)
                end_atom.SetAtomicNum(1)
                idxs.append((end_idx,begin_idx))
            elif begin_symbol == '*' or end_symbol == 'Du':
                new_distance = atom_constants.BONDING_DISTANCES['H',\
                                                                end_symbol,str(bond_type).lower()]
                Chem.rdMolTransforms.SetBondLength(frag.molrdkit.GetConformer(),
                                                   begin_idx, end_idx, new_distance)
                begin_atom.SetAtomicNum(1)
                idxs.append((begin_idx,end_idx))
            else:
                raise ValueError('No dummy atoms found')
    Chem.SanitizeMol(frag.molrdkit)
    if verbose: frag.saveToPDB('final_frag')

    return idxs

def _cleanReplaceFrag(output, keep=False):
    outdir = os.path.dirname(output)
    basename = os.path.basename(output)
    os.system('mv %s/out_rep/* %s'%(output,output))
    os.system('rm -r %s/out_rep'%output)
    if not keep:
        os.system('rm %s/frag_prepared.pdb'%output)
        os.system('rm %s/LIG_original.pdb'%output)
        os.system('rm %s/LIG_p.pdb'%output)
        os.system('rm %s/lig_prepared.pdb'%output)
        os.system('rm %s/original_frag.pdb'%output)
        os.system('rm %s/merged.pdb'%output)
        os.system('rm %s/complex_merged.pdb'%output)
