import os 
import sys
import glob
import subprocess
import time

shrodinger_path = '/data/general_software/schrodinger2019-1/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given a directory with .pdb.gz files, extract all the ligands from them')
    parser.add_argument('-d', dest="pdbdir", help = "PDBs directory",required=True)
    args = parser.parse_args()

    pdbdir = args.pdbdir
    
    os.chdir(pdbdir)

    for PDBgz in glob.glob("*.pdb.gz"):
        PDB = PDBgz.replace(".gz","")
        ID = PDB.replace(".pdb","")
        print(PDBgz,PDB,ID)
        cmd1 = "gunzip %s"%PDBgz
        cmd2 = '%srun split_structure.py -m pdb %s %s.pdb -many_files'%(shrodinger_path,PDB,ID)
        print(cmd1)
        os.system(cmd1)
        print(cmd2)
        os.system(cmd2)
