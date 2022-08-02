import argparse
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
import mollib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Analyse, filter and cluster a fragment database of format (ID fragsSMILES)')
    parser.add_argument('-i', dest="infile", help = "Database of fragments")
    parser.add_argument('-o', dest="outfile", help = "Database of fragments")
    parser.add_argument('--fsize', default=30, help='Filter the uniq SMILES database by size')
    parser.add_argument('--hist', default=False, action='store_true', help='Plot a hist of the atom count')
    parser.add_argument('--sim', default=False, action='store_true', help='Cluster by similarity (tanimoto)')
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    fsize = int(args.fsize)
    hist = args.hist
    sim = args.sim
   
    f = open(infile,'r')
    uniq_smile_frags = {}

    totalcount = 0
    errortime = 0
    errorgeneral = 0
    unknown = 0
    erroranchor = 0
    for i,line in enumerate(f):
        if 'Function' in line:
            if 'longer' in line: errortime += 1
            if 'raise' in line: errorgeneral += 1
            continue
        line = line.split()
        ID = line[0]
        if 'unknown' in ID: unknown+=1
        frags = line[-1].split(",")
        totalcount += len(frags)
        for frag in frags:
            if '[*]' not in frag:
                erroranchor += 1
                #print(ID,frag)
                continue
            if frag not in uniq_smile_frags.keys():
                uniq_smile_frags[frag]=[ID]
            else:
                uniq_smile_frags[frag].append(ID)
    f.close()
    print("Number of compounds that couldn't be decomposed due to time: %d"%errortime)
    print("Number of compounds that couldn't be decomposed due to other errors: %d"%errorgeneral)
    print("Number of compounds that weren't decomposed (no anchoring points [*]/A): %d"%erroranchor)
    print("Total number of fragments %d"%(totalcount-erroranchor))
    print("Total number of unique fragments by str(SMILE): %d"%len(uniq_smile_frags.keys()))

    #Generate filtered database by uniq smiles
    funiq = open(outfile+'_uniqSMILE.txt','w',)
    for k in uniq_smile_frags.keys():
        funiq.write(k + " None " + ",".join(uniq_smile_frags[k]) + "\n") #None eqSMILEs
    funiq.close()

    print("------------------------------------------------------------------------------------")

    #Filter the database by size
    frgDB = mollib.MolDB(txtDB=outfile+'_uniqSMILE.txt',paramaters=True)
    sizefilter = 0
    x = []
    keys_to_delete = []
    for i,k in enumerate(frgDB.dicDB.keys()):
        mol = frgDB.dicDB[k][2]
        numatoms = mol.NumAtoms
        x.append(numatoms)
        if numatoms >= fsize:
            sizefilter +=1
            keys_to_delete.append(k)
    
    for k in keys_to_delete:
        del frgDB.dicDB[k]

    print("Number of fragments filtered by size(%d): %d"%(fsize,sizefilter))
    print("DB size after filtering: %d"%len(frgDB.dicDB.keys()))
    if hist:
        plt.hist(x,bins=100,range=(0,320))
        plt.axvline(x=fsize,color = 'red',linestyle='--')
        plt.show()
    frgDB.print_MolDB(output=outfile+'_uniqSMILE_%d'%fsize)

    #Clusterize by similarity
    if sim:
        frgDB.filter_similarity(fingerprint='Morgan4')
        frgDB.print_MolDB(output=outfile+'_uniqSMILE_%d_sim'%fsize)
