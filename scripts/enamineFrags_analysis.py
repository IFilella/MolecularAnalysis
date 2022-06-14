import numpy as np
import mol
import time
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import rdkit.Chem.Lipinski as Lipinski

enamineFra = open('../data/enamineFragments/enamine_fragments.txt','r')

"""
#Get total number of fragments of Chembl Fragment database
count = 0
errorcount = 0
for i,line in enumerate(enamineFra):
    line = line.split()
    try:
        frags = line[-1].split(",")
        if len(line)>2:
            errorcount+=1
            print(line)
    except:
        print(i)
        errorcount+=1
    count+=len(frags)
print('Total number of fragments %d'%count)
print('Total number of errors in %d compounds'%errorcount)
exit()
"""

"""
#Filter Chembl Fragment database by unique smiles
uniq_smile_frags = {}
count = 0
for i, line in enumerate(enamineFra):
    if ('Function' in line) or ('unknownID' in line): continue
    line = line.split()
    ID = line[0]
    frags = line[-1].split(",")
    #print('Unique fragments by smile: %d'%len(uniq_smile_frags.keys()))
    #print('Analized fragments: %d'%count)
    #print(i,ID,frags)
    for frag in frags:
        count += 1
        if frag not in uniq_smile_frags.keys():
            uniq_smile_frags[frag]=[ID]
        else:
            uniq_smile_frags[frag].append(ID)
f = open('/Users/ifilella/BSC/BRICS/data/enamineFragments/enamine_fragments_uniqSMILE.txt','w')
for k in uniq_smile_frags.keys():
    f.write(k + " " + ",".join(uniq_smile_frags[k]) + "\n")
f.close()
exit()
"""

"""
#Analyse Chembl Fragment database by unique smiles
enamineFra_uniqSMILE = open('/Users/ifilella/BSC/BRICS/data/enamineFragments/enamine_fragments_uniqSMILE.txt','r')
uniq_smile_frags = {}
counts = {}
for i, line in enumerate(enamineFra_uniqSMILE):
    line = line.split()
    SMILE = line[0]
    IDs = line[-1]
    if SMILE not in uniq_smile_frags.keys():
        uniq_smile_frags[SMILE]=IDs
    else:
        print("Error")
    #print(line)
    #print(SMILE)
    #print(IDs)
    #print(len(IDs))
    if len(IDs) not in counts.keys():
        counts[len(IDs)]=1
    else:
        counts[len(IDs)]+=1
    #print(counts)
#print(counts)
#print(counts[1])
#kcounts=list[counts.keys()]
#kcounts = np.fromiter(counts.keys(), dtype=int)
#ind = np.where(kcounts<25)
#print(ind)
#print(kcounts[ind])
exit()
"""
"""
#Filter by fragment size
counts = {}
kekuleerror = 0
sizefilter = 0
x = []
enamineFra_uniqSMILE = open('../data/enamineFragments/enamine_fragments_uniqSMILE.txt','r')
enamineFra_uniqSMILE_40 = open('../data/enamineFragments/enamine_fragments_uniqSMILE_40.txt','w')
for i, line in enumerate(enamineFra_uniqSMILE):
    line = line.split()
    SMILE = line[0]
    IDs = line[-1]
    m1 = mol.mol(smile=SMILE)
    try:
        numatoms = m1.mol.GetNumAtoms()
        x.append(numatoms)
    except:
        kekuleerror +=1
        continue
    if numatoms not in counts.keys():
        counts[numatoms]=1
    else:
        counts[numatoms]+=1
    if numatoms >= 40:
        sizefilter +=1
    else:
        enamineFra_uniqSMILE_40.write(SMILE + " " + IDs + "\n")

enamineFra_uniqSMILE.close()
print(kekuleerror)
print(sizefilter)
print(max(x))
plt.hist(x,bins=100,range=(0,320))
plt.axvline(x=40,color = 'red',linestyle='--')
plt.show()
enamineFra_uniqSMILE_40.close()
exit()
"""

#Clusterize Chembl Fragment database by similarity
verbose = True
enamineFra_uniqSMILE_40 = open('../data/enamineFragments/enamine_fragments_uniqSMILE_40.txt','r')
enamineFra_uniqSMILE_40_sim = open('../data/enamineFragments/enamine_fragments_uniqSMILE_40_sim.txt','w')
uniquefrags = {}
count = 0
#Loop through all fragments of enamineFra_uniqSMILE_40db
for i, line in enumerate(enamineFra_uniqSMILE_40):
    count+=1
    line = line.split()
    SMILE = line[0]
    IDs = line[-1]
    m1 = mol.mol(smile=SMILE)
    m1_atoms = m1.mol.GetNumAtoms()
    m1_NOcount = Lipinski.NOCount(m1.mol)
    m1_NHOHcount = Lipinski.NHOHCount(m1.mol)
    m1_rings = Lipinski.RingCount(m1.mol)
    m1_sp3 = Lipinski.FractionCSP3(m1.mol)
    #m1_AliphaticCarbocycles = Lipinski.NumAliphaticCarbocycles(m1.mol)
    #m1_NumAliphaticHeterocycles = Lipinski.NumAliphaticHeterocycles(m1.mol)
    m1_NumAliphaticRings = Lipinski.NumAliphaticRings(m1.mol)
    #m1_NumAromaticCarbocycles = Lipinski.NumAromaticCarbocycles(m1.mol)
    #m1_NumAromaticHeterocycles = Lipinski.NumAromaticHeterocycles(m1.mol)
    m1_NumAromaticRings = Lipinski.NumAromaticRings(m1.mol)
    #Save the first fragment into the dicctionary
    if len(uniquefrags.keys()) == 0:
        #uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_AliphaticCarbocycles,m1_NumAliphaticHeterocycles,m1_NumAliphaticRings,m1_NumAromaticCarbocycles,m1_NumAromaticHeterocycles,m1_NumAromaticRings]
        uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
    #Loop through the unique fragments of the dicctionary
    for j,k in enumerate(uniquefrags.keys()):
        m2 = uniquefrags[k][2]
        m2_atoms = uniquefrags[k][3]
        m2_NOcount = uniquefrags[k][4]
        m2_NHOHcount = uniquefrags[k][5]
        m2_rings = uniquefrags[k][6]
        m2_sp3 = uniquefrags[k][7]
        #m2_AliphaticCarbocycles = uniquefrags[k][8]
        #m2_NumAliphaticHeterocycles = uniquefrags[k][9]
        m2_NumAliphaticRings = uniquefrags[k][8]
        #m2_NumAromaticCarbocycles = uniquefrags[k][11]
        #m2_NumAromaticHeterocycles = uniquefrags[k][12]
        m2_NumAromaticRings = uniquefrags[k][9]
        #To avoide unecessary similarity calculations check if the new frag (m1) and the unique frag (m2)
        #have identical properties susch as the number of atoms or the number of rings.
        #if m1_atoms != m2_atoms or m1_NOcount != m2_NOcount or m1_rings != m2_rings or m1_NHOHcount != m2_NHOHcount or m1_sp3 != m2_sp3 or m1_AliphaticCarbocycles != m2_AliphaticCarbocycles or m1_NumAliphaticHeterocycles != m2_NumAliphaticHeterocycles or m1_NumAliphaticRings != m2_NumAliphaticRings or m1_NumAromaticCarbocycles != m2_NumAromaticCarbocycles or m1_NumAromaticHeterocycles != m2_NumAromaticHeterocycles or m1_NumAromaticRings != m2_NumAromaticRings:
        if m1_atoms != m2_atoms or m1_NOcount != m2_NOcount or m1_rings != m2_rings or m1_NHOHcount != m2_NHOHcount or m1_sp3 != m2_sp3 or m1_NumAliphaticRings != m2_NumAliphaticRings or  m1_NumAromaticRings != m2_NumAromaticRings:
            #If the last iterated unique frag (m2) doesn't share all properties with m1 then save m1 into the dicc
            if j == len(uniquefrags.keys())-1:
                #uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_AliphaticCarbocycles,m1_NumAliphaticHeterocycles,m1_NumAliphaticRings,m1_NumAromaticCarbocycles,m1_NumAromaticHeterocycles,m1_NumAromaticRings]
                uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                break
            else:
                continue
        #If m1 and m2 share identical properties then check the similarity
        else:
            similarity =  mol.get_MolSimilarity(m1,m2)
            #If the last iterated unique frag (m2) and m1 have a similarity different from 1 then store m1 into the dicc
            if j == len(uniquefrags.keys())-1 and similarity != 1:
                #uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_AliphaticCarbocycles,m1_NumAliphaticHeterocycles,m1_NumAliphaticRings,m1_NumAromaticCarbocycles,m1_NumAromaticHeterocycles,m1_NumAromaticRings]
                uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                break
            #If the last iterated unique frag (m2) and m1 have a similarity of 1 store m1 with m2
            elif j == len(uniquefrags.keys())-1 and similarity == 1:
                print("0: " + uniquefrags[k][0] + " " +SMILE)
                uniquefrags[k][0] += ',' + SMILE
                uniquefrags[k][1] += ',' + IDs
                break
            #If the iterated unique frag (m2) and m1 have a similarity of 1 store m1 with m2
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
    #if i == 48000: break

print("Saving into file")
for k in uniquefrags.keys():
    enamineFra_uniqSMILE_40_sim.write(k + " " + uniquefrags[k][0] + " " +uniquefrags[k][1] + "\n")
enamineFra_uniqSMILE_40_sim.close()
