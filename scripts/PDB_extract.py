import os 
import sys
import glob
import subprocess
import time
import argparse

shrodinger_path = '/data/general_software/schrodinger2019-1/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given a directory with .pdb.gz/pdb files, extract all the ligands from them')
    parser.add_argument('-d', dest="pdbdir", help = "PDBs directory",required=True)
    args = parser.parse_args()

    pdbdir = args.pdbdir
    
    os.chdir(pdbdir)

    PDBs = glob.glob("*.pdb*")
    for PDB in PDBs:
        if '.pdb.gz' in PDB:
            cmd1 = "gunzip %s"%PDB
            os.system(cmd1)
            PDB = PDB.replace('.pdb.gz','.pdb')
            ID = PDB.replace(".pdb","")
        elif '.pdb' in PDB:
            ID = PDB.replace(".pdb","")
        print(PDB,ID)
        cmd2 = '%srun split_structure.py -m pdb %s %s.pdb -many_files'%(shrodinger_path,PDB,ID)
        print(cmd2)
        os.system(cmd2)
