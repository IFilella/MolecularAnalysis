from MolecularAnalysis import mol
from MolecularAnalysis import moldb
from rdkit.Chem.Scaffolds import MurckoScaffold
import rdkit.Chem.BRICS as BRICS
import rdkit.Chem as Chem
import numpy as np
import itertools
import re
import pandas as pd
from collections import Counter

def getScaffold(molobj):
    """
    Get the scaffold of a Mol object
    - molobj: Mol object
    """
    scaffold=MurckoScaffold.GetScaffoldForMol(molobj.molrdkit)
    molobj.scaffold=scaffold
    return scaffold

def get2DFragments(molobj, anchoring=True):
    """
    Get 2D BRICS fragments from a Mol object
    - molobj: Mol object
    - anchiring: If False remove the anchoring symols from the
                 fragment smiles (default True)
    """
    molobj.fragments2D=list(BRICS.BRICSDecompose(molobj.molrdkit))
    if not anchoring:
             molobj.fragments2D=[re.sub("(\[[0-9]+\*\])", "[*]", frag)
                                   for frag in molobj.fragments2D]
    return molobj.fragments2D

def get3DFragments(molobj, centers=[], radius=[], verbose=False,
                   anchoring=True, partialperc=0.5, recomposeperc=0.75):
    """
    Get 3D BRICS fragments from a Mol object. If a list of centers and radius
    are provided the fragments will be reconstructed to fit into this boxes
    - molobj: Mol object
    - centers: list of centers (X,Y,Z) coordinates
    - radius: list of radious in Angstroms
    - verbose: If True get additional details
    - anchoring: If False remove the anchoring symols from the
                 fragment smiles (default True)
    - partialperc: (default 0.5)
    - recomposeperc: (default 0.75)
    """
    newmol2=Chem.FragmentOnBRICSBonds(molobj.molrdkit)
    molobj.fragments3D=Chem.GetMolFrags(newmol2,asMols=True,sanitizeFrags=True)
    _get3DFragmentsConnections(molobj)
    if not centers and not radius:
        pass
    else:
        mol_name=molobj.name
        boxesFragments3D=[[] for x in range(len(centers))]
        for i,center in enumerate(centers):
            if verbose: print(center,radius[i])

            #Check if one or more than one frag are in the box
            idx_frags=[]
            for j, frag in enumerate(molobj.fragments3D):
                perc=_getPercenBox(frag,center,radius[i])
                if perc >= partialperc: idx_frags.append(j)
            if verbose: print(idx_frags)

            #If there are no frags in the box go to the next box
            if len(idx_frags)==0: continue

            #If there is one frag in the box store it
            elif len(idx_frags)==1:
                frag=molobj.fragments3D[idx_frags[0]]
                perc=_getPercenBox(frag,center,radius[i])
                if perc >= recomposeperc:
                    frag_name='S%s_%s'%(str(i+1),mol_name)
                    frag.SetProp("_Name",frag_name)
                    if not anchoring:
                        frag=_rmvDummyAtoms(frag)
                        if frag==-1: continue
                    boxesFragments3D[i].append(frag)

            #If there is more than one frag in the box, combine them and store the new frag
            elif len(idx_frags) > 1:
                #Get submatrix of connections for the fragments in the box
                _connections=molobj.fragments3D_connections[np.ix_(idx_frags,idx_frags)]

                #If one of the frags has no connections with te rest it is saved independently
                idx_noconn_frags=[]
                for _idx in range(len(idx_frags)):
                    idx=idx_frags[_idx]
                    if sum(_connections[_idx])==0:
                        frag=molobj.fragments3D[idx]
                        perc=_getPercenBox(frag,center,radius[i])
                        if perc >= recomposeperc:
                            frag_name='S%s_%s'%(str(i+1),mol_name)
                            frag.SetProp("_Name",frag_name)
                            boxesFragments3D[i].append(frag)
                        idx_noconn_frags.append(idx)

                #New idx_frags of connected frags
                if len(idx_noconn_frags) > 0:
                    idx_frags=list(set(idx_frags)-set(idx_noconn_frags))
                if len(idx_frags)==0:
                    continue
                else:
                    #Combine all fragments into a single Mol object:
                    for j in range(len(idx_frags)):
                        frag0=molobj.fragments3D[idx_frags[j]]
                        if j==0:
                            frag1=molobj.fragments3D[idx_frags[j+1]]
                            combined_frags=Chem.CombineMols(frag0,frag1)
                        elif j < len(idx_frags)-1:
                            frag1=molobj.fragments3D[idx_frags[j+1]]
                            combined_frags=Chem.CombineMols(combined_frags,frag1)

                    #Remove dummy atoms 'R/*' and add missing bounds
                    conformer=combined_frags.GetConformer()
                    coords={}
                    todelete=[]
                    toconnect=[]

                    #Fill list of atoms index to delete (todelete) and atoms index
                    # to connect (toconnect)
                    for idx, atom in enumerate(combined_frags.GetAtoms()):
                        symbol=atom.GetSymbol()
                        position=conformer.GetAtomPosition(idx)
                        key=str(position.x) + str(position.y) + str(position.z)
                        if symbol!='*':
                            if key in coords:
                                todelete.append(coords[key])
                                toconnect.append(idx)
                            else:
                                coords[key]=idx
                        else:
                            if key in coords:
                                todelete.append(idx)
                                if combined_frags.GetAtoms()[coords[key]].GetSymbol()!='*':
                                    toconnect.append(coords[key])
                            else:
                                coords[key]=idx
                    toconnect.sort(reverse=True)
                    if verbose: print(toconnect)

                    #Get mapping between toconnect atom idx (combined_frags)
                    #whole ligand atom idx (self.mol)
                    mapping={}
                    for idx_toconnect in toconnect:
                        atom_toconnect=combined_frags.GetAtoms()[idx_toconnect]
                        coords_toconnect=_getAtomCoords(combined_frags,atom_toconnect)
                        for idx_mol, atom_mol in enumerate(molobj.molrdkit.GetAtoms()):
                            coords_mol=_getAtomCoords(molobj.molrdkit,atom_mol)
                            if np.linalg.norm(coords_toconnect-coords_mol)==0:
                                mapping[idx_toconnect]=idx_mol
                    reversed_mapping={_v: _k for _k, _v in mapping.items()}

                    #Get already existing bonds in combined_frags
                    old_bonds=[]
                    for bond in combined_frags.GetBonds():
                        old_bonds.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
                    if verbose: print(old_bonds)

                    _combined_frags=Chem.EditableMol(combined_frags)

                    #Reconstruct the missing bonds between fragments
                    new_bonds=[]
                    for idx_toconnect in toconnect:
                        if verbose: print('-----------')
                        idx_mol=mapping[idx_toconnect]
                        if verbose: print(idx_toconnect,idx_mol)
                        atom_mol=molobj.molrdkit.GetAtoms()[idx_mol]
                        bonds_atom_mol=atom_mol.GetBonds()
                        idxs_atom_mol=[]
                        bondTypes_atom_mol=[]
                        for bond in bonds_atom_mol:
                            idxs_atom_mol.append(bond.GetBeginAtomIdx())
                            idxs_atom_mol.append(bond.GetEndAtomIdx())
                            bondTypes_atom_mol.extend([bond.GetBondType()]*2)
                        if verbose: print(idxs_atom_mol)
                        for _idx, idx in enumerate(idxs_atom_mol):
                            if idx==idx_mol: continue
                            if str(bondTypes_atom_mol[_idx])=='AROMATIC': continue
                            if idx in mapping.values():
                                idx2_toconnect=reversed_mapping[idx]
                                if verbose: print(idx_toconnect,idx2_toconnect,
                                                  bondTypes_atom_mol[_idx])
                                if (idx_toconnect,idx2_toconnect) not in old_bonds and (idx2_toconnect,idx_toconnect) not in old_bonds and (idx_toconnect,idx2_toconnect) not in new_bonds and (idx2_toconnect,idx_toconnect) not in new_bonds:
                                    new_bonds.append((idx_toconnect,idx2_toconnect))
                                    _combined_frags.AddBond(idx_toconnect,idx2_toconnect,
                                                            bondTypes_atom_mol[_idx])
                    #Delete dummy atoms in todelete
                    todelete.sort(reverse=True)
                    for atom in todelete: _combined_frags.RemoveAtom(atom)

                    #Store the new combined frag
                    new_combined_frags=_combined_frags.GetMol()
                    Chem.SanitizeMol(new_combined_frags)

                    #If clean remove dummy atoms
                    if not anchoring:
                        new_combined_frags=_rmvDummyAtoms(new_combined_frags)
                        if new_combined_frags==-1: continue

                    frag_name='S%s_%s'%(str(i+1),mol_name)
                    new_combined_frags.SetProp("_Name",frag_name)
                    boxesFragments3D[i].append(new_combined_frags)

        if verbose: print(boxesFragments3D)
        molobj.boxesFragments3D=boxesFragments3D

def getMolDB3DFragments(moldbobj, norm=True, outname=None, verbose=False):
    """
    Fragmentate all molecules of a MolDB and compute the frequency of appearence of each fragment
    as well as the frequency of connections between fragment pairs
    - moldbobj: MolDB obj
    - norm: If True return the normalize frequency of fragments instead of the
            count of fragments
    - outname: outname to save the frequencies DataFrames to .csv
    - verbose: If True get additional details
    """
    connection_dfs=[]
    total_frags=[]
    for i,key in enumerate(moldbobj.dicDB.keys()):
        mol=moldbobj.dicDB[key][-1]
        get3DFragments(mol)
        connections=mol.fragments3D_connections
        frags=mol.fragments3D
        canonical_smiles=[]
        for frag in frags:
            canonical_smile=Chem.MolToSmiles(frag, canonical=True)
            canonical_smile=re.sub("(\[[0-9]+\*\])", "[*]", canonical_smile)
            canonical_smiles.append(canonical_smile)
        total_frags += canonical_smiles
        connection_df=pd.DataFrame(connections, index=canonical_smiles, columns=canonical_smiles)
        connection_dfs.append(connection_df)

    counter=Counter(total_frags)
    freq_frags=pd.DataFrame({"counter": counter})
    if norm:
        counts=float(sum(freq_frags['counter'].tolist()))
        freq_frags['counter']=freq_frags['counter'] / counts
    moldbobj.fqFrags=freq_frags

    total_frags_unique=np.unique(np.array(total_frags))
    if verbose:
        print('Total number of fragments: %d\nTotal number of unique fragments %d'
              %(len(total_frags),len(total_frags_unique)))
    freq_mat=[[0] * len(total_frags_unique)] * len(total_frags_unique)
    freq_df=pd.DataFrame(freq_mat, index=total_frags_unique, columns=total_frags_unique)

    for k,connection_df in enumerate(connection_dfs):
        indexes=connection_df.index.to_list()
        for i,index1 in enumerate(indexes):
            for j,index2 in enumerate(indexes):
                if j >= i:
                    value=connection_df.loc[index1,index2]
                    if isinstance(value,np.integer):
                        if value==1:
                            freq_df.loc[index1, index2] += 1
                            freq_df.loc[index2, index1] += 1
                    elif isinstance(value,pd.core.frame.DataFrame):
                        if np.count_nonzero(value) > 0:
                            if index1==index2:
                                freq_df.loc[index1, index2] += np.count_nonzero(value)
                            else:
                                freq_df.loc[index1, index2] += np.count_nonzero(value)
                                freq_df.loc[index2, index1] += np.count_nonzero(value)
                    elif isinstance(value,pd.core.series.Series):
                        if not (value==0).all():
                            freq_df.loc[index1, index2] += np.count_nonzero(value)
                            freq_df.loc[index2, index1] += np.count_nonzero(value)
                    else:
                        raise ValueError('Error with dtype of frags_connections')

    if norm:
        indexes=freq_df.index.to_list()
        for index in indexes:
            counts=float(sum(freq_df.loc[index,:].tolist()))
            freq_df.loc[index,:]=freq_df.loc[index,:] / counts

    moldbobj.fqFragsConnections=freq_df
    if outname!=None:
        moldbobj.fqFragsConnections.to_csv('%s_fqFragConnections.csv'%outname, sep=',')
        moldbobj.fqFrags.to_csv('%s_fqFrags.csv'%outname,sep=',')

def _get3DFragmentsConnections(molobj):
    """
    Given a Mol object with its 3D fragments, get a matrix of inter fragment connections.
    - molobj: Mol obj with its 3D fragments
    """
    if not hasattr(molobj, 'fragments3D'):
        raise ValueError('BRICS fragments aren\'t calculated. Run get3DFragments')
    _get_LigToFrag_AtomMapping(molobj)
    nfrags=len(molobj.fragments3D)
    frags_connections=np.zeros((nfrags,nfrags),dtype=int)
    for atom_lig, atoms_frags in molobj.LigToFrag_AtomMapping.items():
        _connections=np.where(np.array(atoms_frags)!=None)[0]
        if len(_connections) > 1:
            connections=list(itertools.combinations(_connections,2))
            for connection in connections:
                frags_connections[connection[0]][connection[1]]=1
                frags_connections[connection[1]][connection[0]]=1
    molobj.fragments3D_connections=frags_connections

def _get_LigToFrag_AtomMapping(molobj):
    """
    Given a Mol object with its 3D fragments and matrix of fragment connections, get
    a dictionary where its keys are the Mol object atom indices and its values a list of
    fragments where these atoms are present and its corresponding atom index in the fragment
    - molobj: Mol obj with its 3D fragments
    """
    mapping={}
    if not hasattr(molobj, 'fragments3D'):
        raise ValueError('BRICS fragments aren\'t calculated. Run get3Dfragments')
    nfrags=len(molobj.fragments3D)
    ligconformer=molobj.molrdkit.GetConformer()
    lig_atom_coords={}
    for atom in molobj.molrdkit.GetAtoms():
        atom_idx=atom.GetIdx()
        atom_coords=_getAtomCoords(molobj.molrdkit,atom)
        lig_atom_coords[atom_idx]=atom_coords
        mapping[atom_idx]=[None] * nfrags
    for i,frag in enumerate(molobj.fragments3D):
        fragconformer=frag.GetConformer()
        for atom_frag in frag.GetAtoms():
            atom_frag_idx=atom_frag.GetIdx()
            atom_frag_coords=_getAtomCoords(frag,atom_frag)
            for atom_lig_idx, atom_lig_coords in lig_atom_coords.items():
                dist=np.linalg.norm(atom_frag_coords - atom_lig_coords)
                if dist==0:
                    mapping[atom_lig_idx][i]=atom_frag_idx
    molobj.LigToFrag_AtomMapping=mapping

def _rmvDummyAtoms(molrdkit):
    """
    Remove dummy atoms from an rdkit mol object
    - molrdkit: RDKit mol object
    """
    _molrdkit=Chem.EditableMol(molrdkit)
    todelete=[]
    for idx,atom in enumerate(molrdkit.GetAtoms()):
        atomtype=atom.GetSymbol()
        if atomtype=='*': todelete.append(idx)
    todelete.sort(reverse=True)
    for idx in todelete: _molrdkit.RemoveAtom(idx)
    new_molrdkit=_molrdkit.GetMol()
    try:
        Chem.SanitizeMol(new_molrdkit)
    except:
        print('ERROR in %s'%self.name)
        return -1
    return new_molrdkit

def _subsDummyAtomsMolDB(moldbobj,verbose=False):
    """
    Given a MolDB object substitute the dummy atoms for atoms with the corresponding valence
    for each molecule
    - moldbobj: MolDB object
    - verbose: If True get additional details
    """
    kekuleerror1=0
    kekuleerror2=0
    SMILES=list(moldbobj.dicDB.keys())
    totalsmiles=len(SMILES)
    for i,SMILE in enumerate(SMILES):
        if verbose: print(str(i+1) + "/" + str(totalsmiles))
        eqSMILES=moldbobj.dicDB[SMILE][0].split(',')
        IDs=moldbobj.dicDB[SMILE][1]

        #Check SMILE
        new_SMILE=''
        auxmol=Mol(smile=SMILE)
        auxerror=_subsDummyAtomsMol(auxmol)

        #Check eqSMILES and define new_eqSMILES
        new_eqSMILES=[]
        for eqSMILE in eqSMILES:
            auxmol2=Mol(smile=eqSMILE)
            auxerror2=_subsDummyAtomsMol(auxmol2)
            if auxerror2:
                new_eqSMILES.append(auxmol2.smile)

        #Define new_SMILE and count errors
        if auxerror:
            new_SMILE=auxmol.smile
        else:
            if len(new_eqSMILES) > 1:
                new_SMILE=new_eqSMILES[0]
                kekuleerror2+=1
            else:
                kekuleerror1+=1

        #Modify dicDB
        del moldbobj.dicDB[SMILE]
        if new_SMILE!='':
            new_eqSMILES=','.join(new_eqSMILES)
            moldbobj.dicDB[new_SMILE]=[new_eqSMILES,IDs,auxmol]

        i+=1
        if verbose:
            print('-----------------------------------------------')
    if verbose:
        print('Total analysed smiles: %d'%totalsmiles)
        print('Total modified smiles: %d'%(totalsmiles-kekuleerror1-kekuleerror2))
        print('SMILES with kekule error substituted by an eqSMILE: %d'%kekuleerror2)
        print('SMILES with a kekule error without an eqSMULE: %d'%kekuleerror1)

def _subsDummyAtomsMol(molobj):
    """
    Given a Mol object substitute its dummy atoms for atoms with the corresponding valence
    - molobj: Mol object
    """
    print('Old SMILE: ' + molobj.smile)
    old_smile=molobj.smile
    #Get indeces of the anchoring points in the smile:
    count=0
    new_smile=old_smile
    while new_smile.startswith('[*].'): new_smile=new_smile[4:]
    while new_smile.endswith('.[*]'): new_smile=new_smile[:-4]
    indices=[i for i, c in enumerate(new_smile) if c=='*']
    new_smile=list(new_smile)
    for atom in molobj.molrdkit.GetAtoms():
        if atom.GetSymbol()=='*' and count < len(indices):
            valence=atom.GetExplicitValence()
            if valence==0:
                new_smile[indices[count]]=''
                new_smile[indices[count]-1]=''
                new_smile[indices[count]+1]=''
            elif valence==1:
                if atom.GetIsAromatic():
                    new_smile[indices[count]]='h'
                else:
                    new_smile[indices[count]]='H'
                count+=1
            elif valence > 1 and valence < 5:
                if atom.GetIsAromatic():
                    new_smile[indices[count]]='c'
                else:
                    new_smile[indices[count]]='C'
                count+=1
            elif valence==5:
                if atom.GetIsAromatic():
                    new_smile[indices[count]]='p'
                else:
                    new_smile[indices[count]]='P'
                count+=1
            elif valence==6:
                if atom.GetIsAromatic():
                    new_smile[indices[count]]='s'
                else:
                    new_smile[indices[count]]='S'
                count+=1
            elif valence > 6:
                return False
            else:
                raise ValueError('The anchoring point %d (*) of %s have a valence %d greater than 4.%s'%(count,old_smile,valence,''.join(new_smile)))
    new_smile=''.join(new_smile)
    molobj.smile=new_smile
    try:
        molobj.molrdkit=Chem.MolFromSmiles(molobj.smile)
        if molobj.molrdkit==None:
            print('Kekulize ERROR')
            return False
        else:
            print('New Smile: ' + molobj.smile)
            return True
    except:
        print('Kekulize ERROR')
        return False

def _getAtomCoords(molrdkit,atomrdkit):
    """
    Given a RDKit mol object and an RDKit atom object of the mol object compute the atom coords
    - molrdkit: RDKit mol object
    - atomrdkit: RDKit atom object
    """
    conformer=molrdkit.GetConformer()
    atom_idx=atomrdkit.GetIdx()
    position=conformer.GetAtomPosition(atom_idx)
    atom_coords=np.array((position.x, position.y, position.z))
    return atom_coords

def _getPercenBox(molrdkit,center,radius):
    """
    Given a RDKit mol object and a box defined with a center and a radious get the percentage
    of atoms of the mol inside the box
    - molrdkit: RDKit mol object
    - center: center in [X,Y,Z]
    - radius: radius in Angstroms
    """
    total=len(molrdkit.GetAtoms())
    count=0
    for i, atom in enumerate(molrdkit.GetAtoms()):
        coords=_getAtomCoords(molrdkit,atom)
        if np.linalg.norm(coords-center) <= radius:
            count +=1
    return count/total
