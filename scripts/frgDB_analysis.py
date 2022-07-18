import argparse
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
import mol

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
        funiq.write(k + " " + ",".join(uniq_smile_frags[k]) + "\n")
    funiq.close()

    print("------------------------------------------------------------------------------------")

    #Filter the database by size
    x = []
    kekuleerror = 0
    sizefilter = 0
    funiq = open(outfile+'_uniqSMILE.txt','r')
    filesize = open(outfile+'_uniqSMILE_%d.txt'%fsize,'w')
    for i,line in enumerate(funiq):
        line = line.split()
        SMILE = line[0]
        IDs = line[-1]
        m1 = mol.mol(smile=SMILE)
        try:
            numatoms = m1.mol.GetNumAtoms()
            x.append(numatoms)
        except:
             kekuleerror += 1
             continue
        if numatoms > fsize:
            sizefilter +=1
        else:
            filesize.write(SMILE + " " + IDs + "\n")
    print("Number of fragments discarded due to error while computing the number of atoms (kekuleerror): %d"%kekuleerror)
    print("Number of fragments filtered by size(%d): %d"%(fsize,sizefilter))
    if hist:
        plt.hist(x,bins=100,range=(0,320))
        plt.axvline(x=fsize,color = 'red',linestyle='--')
        plt.show()
    funiq.close()
    filesize.close()
 
    print("------------------------------------------------------------------------------------")

    #Clusterize by similarity
    if sim:
        filesize = outfile+'_uniqSMILE_' + str(fsize) + '.txt'
        filesim = outfile+'_uniqSMILE_' + str(fsize) + '_simclst.txt'
        mol.filter_db_similarity(filesize,filesim,verbose=True)
