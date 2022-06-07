import mol
import numpy as np
import warnings
import multiprocessing as mp
import time
from pebble import ProcessPool
from concurrent.futures import TimeoutError

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
    infile = '/Users/ifilella/BSC/BRICS/data/chemblCompounds/chembl_30_chemreps_500.txt'
    chemblCompounds = open(infile,'r')
    chemblFragments = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_30_frags.txt','w')
    chemblFragments.write('original_chembl_id fragment_smile')

    with ProcessPool(max_workers= mp.cpu_count(),max_tasks=10) as pool:
        for i,line in enumerate(chemblCompounds):
            if i<=0: continue
            future = pool.schedule(get_fragments,args=[line],timeout=30)
            future.add_done_callback(task_done)
