import numpy as np
import warnings
import multiprocessing as mp
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import argparse
import sys
sys.path.insert(1, '../')
import mol

def get_fragments(line):
    line = line.replace("\n","").split("\t")
    ID = line[0]
    SMILE = line[1]
    InChI = line[2]
    m = mol.mol(InChI=InChI)
    m.get_BRICSdecomposition()
    m.get_clean_fragments()
    res = '%s %s'%(ID,','.join(m.cfragments)) 
    return ID,res

def task_done(future):
    try:
        result = future.result()
        print(result[1])
    except TimeoutError as error:
        print('Function took longer than %d seconds'%error.args[1])
    except Exception as error:
        print('Function raised %s'%error)
        print(error.traceback)  # traceback of the function

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Decompose a database of compounds in the format: (ID SMILE InChI InChI_key) into a database of fragments in the format: (ID Fragments(,)). The process is run in parallel discarding compounds which take more than maxtime to decompose")
    parser.add_argument('-i', dest="infile", help = "Database of compounds")
    parser.add_argument('--maxtime',dest='maxtime',help="maximum time to decompose a compound into their fragments",default=30)
    args = parser.parse_args()
    infile = args.infile
    maxtime = args.maxtime
    
    chemblCompounds = open(infile,'r')

    with ProcessPool(max_workers= mp.cpu_count()) as pool:
        for i,line in enumerate(chemblCompounds):
#            if i<=0: continue
            future = pool.schedule(get_fragments,args=[line],timeout=maxtime)
            future.add_done_callback(task_done)
