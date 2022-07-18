import numpy as np
import warnings
import multiprocessing as mp
import argparse
import sys
sys.path.insert(1, '../')
import mol

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Decompose a database of compounds in the format: (ID SMILE InChI InChI_key) into a database of fragments in the format: (ID Fragments(,)). The process is run in parallel.")
    parser.add_argument('-i', dest="infile", help = "Database of compounds")
    parser.add_argument('-o', dest="outfile", help = "Output name for the database of fragmenets")
    args = parser.parse_args()
    infile = args.infile
    chemblCompounds = open(infile,'r')
    chemblFragments = open(outfile,'w')
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
