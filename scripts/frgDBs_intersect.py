import argparse
import sys
sys.path.insert(1, '../')
import mollib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given two fragment databases with the format (SMILE simSMILES IDs), intersect them')
    parser.add_argument('-db1', dest="db1", help = "First database of fragments as txt file")
    parser.add_argument('-db2', dest="db2", help = "Second database of fragments as txt file")
    parser.add_argument('-t', dest="simt", default = 1, help = "Similariy threshold")
    parser.add_argument('-o', dest="out", default = None, help = "To output the intersected molDB")
    args = parser.parse_args()
    fdb1 = args.db1
    fdb2 = args.db2
    simt = float(args.simt)
    out = args.out

    db1 = mollib.MolDB(txtDB=fdb1,paramaters=True,verbose=True)
    #db1.save_MolDB(fdb1.replace('.txt',''))
    db2 = mollib.MolDB(txtDB=fdb2,paramaters=True,verbose=True) 
    #db2.save_MolDB(fdb2.replace('.txt',''))
    #db1 = mollib.MolDB(dicDB=fdb1.replace('.txt','.p'),paramaters=True,verbose=True)
    #db2 = mollib.MolDB(dicDB=fdb2.replace('.txt','.p'),paramaters=True,verbose=True)
    
    db3 = mollib.intersect_MolDBs(db1,db2,simt,out)
