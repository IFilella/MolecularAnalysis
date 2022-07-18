import argparse
import rdkit.Chem.BRICS as BRICS
import re
from pebble import ProcessPool
import multiprocessing as mp
import warnings
import concurrent.futures
import sys
sys.path.insert(1, '../')
import mol

def get_fragments(cpd,ID):
    fragments = list(BRICS.BRICSDecompose(cpd))
    cfragments = [re.sub("(\[.*?\])", "[*]", frag) for frag in fragments]
    res = '%s %s'%(ID,','.join(cfragments))
    return ID,res

def task_done(future):
    try:
        result = future.result()
        print(result[1])
    except concurrent.futures.TimeoutError as error:
        print("Function took longer than %.3f seconds"%error.args[1])
    except Exception as error:
        print('Function raised error')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Decompose a sdf database of compounds into a database of fragments in the format: (ID Fragments(,)). The process is run in parallel discarding compounds which take more than maxtime to decompose")
    parser.add_argument('-i', dest="infile", help = "Database of compounds")
    parser.add_argument('--maxtime',dest='maxtime',help="maximum time to decompose a compound into their fragments",default=30)
    args = parser.parse_args()
    infile = args.infile
    maxtime = float(args.maxtime)
    
    cpdDB = mol.read_compoundDB(infile)

    #mols = [mol for mol in cpdDB]
    #print('Total compounds: %s'%str(len(mols)))

    with ProcessPool(max_workers= mp.cpu_count(),max_tasks=0) as pool:
        for cpd in cpdDB:
            try:
                ID = cpd.GetProp("idnumber")
            except:
                ID = 'unknownID'
            future = pool.schedule(get_fragments,args=[cpd,ID],timeout=maxtime)
            future.add_done_callback(task_done)
