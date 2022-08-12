from prody import *
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given a PDB list from https://www.rcsb.org/ download all associated .pdb.gz files')
    parser.add_argument('-l', dest="pdblist", help = "List of PDBs",required=True)
    parser.add_argument('-o', dest="outdir", help = "Output directory to store the PDBs",default=None)
    args = parser.parse_args()

    #Parse inputs
    pdblist = args.pdblist
    PDB_IDs = np.genfromtxt(pdblist,dtype=str,delimiter=',')
    outdir = args.outdir
    if outdir != None:
        try:
            os.chdir(outdir)
        except:
            os.makedirs(outdir)
            os.chdir(outdir)
    else:
        if os.path.isdir('dow_PDBs/'):
            os.chdir('dow_PDBs/')
        else:
            os.makedirs('dow_PDBs/')
            os.chdir('dow_PDBs/')

    print('There are %d PDBs to download'%len(PDB_IDs))

    #Download the PDBs
    for PDB_ID in PDB_IDs:
        print(PDB_ID)
        pdb = parsePDB('%s'%(PDB_ID))
