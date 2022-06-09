import numpy as np
import mol
import time
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import rdkit.Chem.Lipinski as Lipinski

chemblFra = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments.txt','r')

"""
#Get total number of fragments of Chembl Fragment database
count = 0
for i,line in enumerate(chemblFra):
    line = line.split()
    frags = line[-1].split(",")
    count+=len(frags)
print('Total number of fragments %d'%count)
"""

"""
#Filter Chembl Fragment database by unique smiles
uniq_smile_frags = {}
count = 0
for i, line in enumerate(chemblFra):
    if 'Function' in line: continue
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
f = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE.txt','w')
for k in uniq_smile_frags.keys():
    f.write(k + " " + ",".join(uniq_smile_frags[k]) + "\n")
f.close()
exit()
"""

"""
#Analyse Chembl Fragment database by unique smiles
chemblFra_uniqSMILE = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE.txt','r')
uniq_smile_frags = {}
counts = {}
for i, line in enumerate(chemblFra_uniqSMILE):
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
chemblFra_uniqSMILE = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE.txt','r')
chemblFra_uniqSMILE_40 = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE_40.txt','w')
for i, line in enumerate(chemblFra_uniqSMILE):
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
        chemblFra_uniqSMILE_40.write(SMILE + " " + IDs + "\n")

print(kekuleerror)
print(sizefilter)
plt.hist(x,bins=100,range=(0,320))
plt.axvline(x=40,color = 'red',linestyle='--')
plt.show()
chemblFra_uniqSMILE_40.close()
exit()
"""

#Clusterize Chembl Fragment database by similarity
verbose = True
chemblFra_uniqSMILE_40 = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE_40.txt','r')
chemblFra_uniqSMILE_40_sim = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_fragments_uniqSMILE_40_sim.txt','w')
uniquefrags = {}
count = 0
for i, line in enumerate(chemblFra_uniqSMILE_40):
    count+=1
    line = line.split()
    SMILE = line[0]
    IDs = line[-1]
    m1 = mol.mol(smile=SMILE)
    m1_atoms = m1.mol.GetNumAtoms()
    m1_NOcount = Lipinski.NOCount(m1.mol)
    #Only for the first line
    if len(uniquefrags.keys()) == 0:
        uniquefrags[SMILE] = [SMILE,IDs,m1]
    for j,k in enumerate(uniquefrags.keys()):
        m2 = uniquefrags[k][2]
        m2_atoms = m2.mol.GetNumAtoms()
        m2_NOcount = Lipinski.NOCount(m2.mol)
        #First check the number of atoms of m1 and m2 to avoid unecessary calculations
        #if m1_atoms != m2_atoms:
        #    if j == len(uniquefrags.keys())-1:
        #        uniquefrags[SMILE] = [SMILE,IDs,m1]
        #        break
        #    else:
        #        continue
        ##Second check the number of N and O to avoid unecessary calculations
        #elif m1_NOcount != m2_NOcount:
        #    if j == len(uniquefrags.keys())-1:
        #        uniquefrags[SMILE] = [SMILE,IDs,m1]
        #        break
        #    else:
        #        continue
        ##If the number of atoms of m1 and m2 the same compute its similarity
        #else:
        if True:
            similarity =  mol.get_MolSimilarity(m1,m2)
            if j == len(uniquefrags.keys())-1 and similarity != 1:
                uniquefrags[SMILE] = [SMILE,IDs,m1]
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
    if i == 2400: break

print("Saving into file")
for k in uniquefrags.keys():
    chemblFra_uniqSMILE_40_sim.write(k + " " + uniquefrags[k][0] + " " +uniquefrags[k][1] + "\n")
chemblFra_uniqSMILE_40_sim.close()
