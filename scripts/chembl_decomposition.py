import mol
import numpy as np
import warnings
import multiprocessing as mp
import time
from pebble import process, TimeoutError

fn = 'temp.txt'

def get_fragments(i,line,q):
    line = line.replace("\n","").split("\t")
    ID = line[0]
    SMILE = line[1]
    InChI = line[2]
    m = mol.mol(InChI=InChI)
    m.get_BRICSdecomposition()
    m.get_clean_fragments()
    res = '%d %s %s'%(i, ID,','.join(m.cfragments))
    print(i, ID)
    q.put(res)
    return res

def listener(q):
	with open(fn, 'w') as f:
		while 1:
			m = q.get()
			if m == 'kill':
				f.write('killed')
				break
			f.write(str(m) + '\n')
			f.flush()

"""
for i,line in enumerate(chemblCompounds):
    line = line.replace("\n","").split("\t")
    if len(line) == 4 and i > 0:
        ID = line[0]
        SMILE = line[1]
        InChI = line[2]
        m = mol.mol(InChI=InChI)
        m.get_BRICSdecomposition()
        m.get_clean_fragments()
        print(i, ID, m.cfragments)
        for frag in m.cfragments:
            chemblFragments.write('%s %s\n'%(ID,frag))

    else:
        warnings.warn(f'There are missing data')
"""
if __name__ == '__main__':
    infile = '/Users/ifilella/BSC/BRICS/data/chemblCompounds/chembl_30_chemreps_5000.txt'
    chemblCompounds = open(infile,'r')
    chemblFragments = open('/Users/ifilella/BSC/BRICS/data/chemblFragments/chembl_30_frags.txt','w')
    chemblFragments.write('original_chembl_id fragment_smile')

    manager = mp.Manager()
    q = manager.Queue()
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count()+2)

    #Put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #Fire off workers
    jobs= []
    for i,line in enumerate(chemblCompounds):
        if i <= 0: continue
        job = pool.apply_async(get_fragments, (i,line,q))
        jobs.append(job)

    for job in jobs:
        job.get()
    

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()
