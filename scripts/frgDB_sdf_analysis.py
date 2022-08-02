import argparse
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../')
import mollib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Transform a sdf fragment database to a database of format (ID SMILE) and then analyse, filter and cluster it.')
    parser.add_argument('-i', dest="infile", help = "Database of fragments")
    parser.add_argument('-o', dest="outfile", help = "Database of fragments")
    parser.add_argument('--fsize', default=30, help='Filter the uniq SMILES database by size')
    parser.add_argument('--hist', default=False, action='store_true', help='Plot a hist of the atom count')
    parser.add_argument('--sim', default=False, action='store_true', help='Cluster by similarity (tanimoto)')
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    fsize = args.fsize
    fsize = int(args.fsize)
    hist = args.hist
    sim = args.sim

    cpdDB = mollib.MolDB(sdfDB = infile, paramaters=True)

    #Change the format of the database and filter by uniqSMILE
    cpdDB.print_MolDB(output=outfile+'_uniqSMILE')
    print("Unique SMILES: %d"%len(cpdDB.dicDB.keys()))

    #Filter the database by size
    x = []
    sizefilter = 0
    filesize = open(outfile+'_uniqSMILE_%d.txt'%fsize,'w')
    keys_to_delete = []
    for i,k in enumerate(cpdDB.dicDB.keys()):
        mol = cpdDB.dicDB[k][2]
        mol.get_NumAtoms()
        numatoms = mol.NumAtoms
        x.append(numatoms)
        if numatoms >= fsize:
            sizefilter +=1
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del cpdDB.dicDB[k]
    print("Fragments filtered by size: %d"%sizefilter)
    print("DB size after filtering: %d"%len(cpdDB.dicDB.keys()))
    cpdDB.print_MolDB(output=outfile+'_uniqSMILE_%d'%fsize)

    if hist:
        plt.hist(x,bins=100,range=(0,320))
        plt.axvline(x=fsize,color = 'red',linestyle='--')
        plt.show()
    
    if sim:
        cpdDB.filter_similarity(fingerprint='Morgan4')
        cpdDB.print_MolDB(output=outfile+'_uniqSMILE_%d_sim'%fsize)
