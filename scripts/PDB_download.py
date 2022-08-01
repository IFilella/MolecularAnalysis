from prody import *
import numpy as np
import argparse

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description ='Given a PDB list from https://www.rcsb.org/ download all associated .pdb.gz files')
     parser.add_argument('-l', dest="pdblist", help = "List of PDBs",required=True)
     parser.add_argument('-o', dest="outdir", help = "Output directory to store the .pdb.gz files",required=True)
     args = parser.parse_args()

     pdblist = args.pdblist
     outdir = args.outdir

    PDB_IDs = np.genfromtxt(pdblist,dtype=str,delimiter=',')
    print('There are %d PDBs to download'%len(PDB_IDs))

    #Download the PDBs
    for PDB_ID in PDB_IDs:
        print(PDB_ID)
        pdb = parsePDB('%s/%s.pdb.gz'%(outdir,PDB_ID))
