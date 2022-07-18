import argparse
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
import mol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Transform a sdf fragment database to a database of format (ID SMILE) and then analyse, filter and cluster it.')
    parser.add_argument('-i', dest="infile", help = "Database of fragments")
    parser.add_argument('-o', dest="outfile", help = "Database of fragments")
    parser.add_argument('--uniq', default=False, action='store_true', help='Filter the database by uniq SMILES')
    parser.add_argument('--fsize', default=30, help='Filter the uniq SMILES database by size')
    parser.add_argument('--hist', default=False, action='store_true', help='Plot a hist of the atom count')
    parser.add_argument('--sim', default=False, action='store_true', help='Cluster by similarity (tanimoto)')
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    uniq = args.uniq
    fsize = args.fsize
    hist = args.hist
    sim = args.sim

    cpdDB = mol.read_compoundDB(infile)
    
    f = open(outfile+'.txt','w')
    uniq_smile_frags = {}

    #Change the format of the database
    for i,frag in enumerate(cpdDB):
        SMILE = Chem.MolToSmiles(frag)
        ID = frag.GetProp("Catalog ID")
        f.write('%s %s\n'%(ID,SMILE))
        if uniq:
            if SMILE not in uniq_smile_frags.keys():
                uniq_smile_frags[SMILE]=[ID]
            else:
                print(SMILE,ID,uniq_smile_frags[SMILE])
                uniq_smile_frags[SMILE].append(ID)
    f.close()

    #Filter the database by uniq smiles
    if uniq:
        funiq = open(outfile+'_uniqSMILE.txt','w',)
        for k in uniq_smile_frags.keys():
            funiq.write(k + " " + ",".join(uniq_smile_frags[k]) + "\n")
        funiq.close()
        print("Unique SMILES: %d"%len(uniq_smile_frags.keys()))

    #Filter the database by size
    if uniq:
        x = []
        kekuleerror = 0
        sizefilter = 0
        funiq = open(outfile+'_uniqSMILE.txt','r')
        filesize = open(outfile+'_uniqSMILE_40.txt','w')
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
            if numatoms >= fsize:
                sizefilter +=1
            else:
                filesize.write(SMILE + " " + IDs + "\n")
        print("Error while computing the number of atoms (kekuleerror): %d"%kekuleerror)
        print("Fragments filtered by size: %d"%sizefilter)
        if hist:
            plt.hist(x,bins=100,range=(0,320))
            plt.axvline(x=40,color = 'red',linestyle='--')
            plt.show()
        funiq.close()
        filesize.close()
    
    if sim:
        filesize = outfile+'_uniqSMILE_40.txt'
        filesim = outfile+'_uniqSMILE_40_sim.txt'
        mol.filter_db_similarity(filesize,filesim,verbose=False)
