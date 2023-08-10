import numpy as np
import mollib
import time
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import rdkit.Chem.Lipinski as Lipinski

verbose = False
chemblFra_uniqSMILE_40 = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE_40.txt','r')

uniquefrags = {}
count = 0
for i, line in enumerate(chemblFra_uniqSMILE_40):
    count+=1
    line = line.split()
    SMILE = line[0]
    IDs = line[-1]
    m1 = mollib.Mol(smile=SMILE)
    m1_atoms = m1.mol.GetNumAtoms()
    m1_NOcount = Lipinski.NOCount(m1.mol)
    m1_NHOHcount = Lipinski.NHOHCount(m1.mol)
    m1_rings = Lipinski.RingCount(m1.mol)
    print(i)
    #Only for the first line
    if len(uniquefrags.keys()) == 0:
        uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings]
    for j,k in enumerate(uniquefrags.keys()):
        m2 = uniquefrags[k][2]
        m2_atoms = uniquefrags[k][3]
        m2_NOcount = uniquefrags[k][4]
        m2_NHOHcount = uniquefrags[k][5]
        m2_rings = uniquefrags[k][6]
        similarity =  mollib.get_MolSimilarity(m1,m2)
        if m1_atoms != m2_atoms or m1_NOcount != m2_NOcount:
            if similarity == 1 and  m1_NOcount != m2_NOcount and m1_atoms == m2_atoms:
                print(uniquefrags[k][0] + " " +SMILE)
            if j == len(uniquefrags.keys())-1:
                uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings]
                break
            else:
                continue
        else:
            if j == len(uniquefrags.keys())-1 and similarity != 1:
                uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings]
                break
            elif j == len(uniquefrags.keys())-1 and similarity == 1:
                #print("0: " + uniquefrags[k][0] + " " +SMILE)
                uniquefrags[k][0] += ',' + SMILE
                uniquefrags[k][1] += ',' + IDs
                break
            elif similarity == 1:
                #print("1: " + uniquefrags[k][0] + " " +SMILE)
                uniquefrags[k][0] += ',' + SMILE
                uniquefrags[k][1] += ',' + IDs
                break
            else:
                continue
    if verbose:
        print('Unique fragments: %d'%len(uniquefrags.keys()))
        print('Analized fragments: %d'%count)
    if i == 48000: break
