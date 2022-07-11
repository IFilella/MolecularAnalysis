import mol
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given a list of fragment databases with the format (SMILE simSMILES IDs), plot them into a mapping such as t-SNE or UMAP or PCA')
    parser.add_argument('-dbs', dest="dbs", nargs = '+', help = "List of databases of fragments",required=True)
    parser.add_argument('-m', dest="mapping", default = 'tSNE', help = "Mapping method",required=True)
    parser.add_argument('-o', dest="out", default = None, help = "Map output name")    
    parser.add_argument('--txt',default=False,action='store_true',help='To pass either a txt moldb (if True) or a pickle moldb (if False)')
    parser.add_argument('-del', dest="delt", default = None, help = "Delimiter for the database name")
    parser.add_argument('-r', dest= "random_max", default = 5000, help = "Randomly select X elements of each databse")
    args = parser.parse_args()
    fdbs = args.dbs
    mapping = args.mapping
    if mapping not in ['tSNE','UMAP','PCA']:
        raise ValueError('%s is not a valid mapping method. Mapping must be either t-SNE, UMAP or PCA.'%mapping)
    out = args.out
    txt = args.txt
    delimiter = args.delt
    random_max = int(args.random_max)

    dbs = []
    names = []
    for fdb in fdbs:
        if txt:
            db = mol.molDB(txtDB=fdb,paramaters=True,verbose=True)
            db.save_molDB(fdb.replace('.txt',''))
        else:
            db = mol.molDB(dicDB=fdb,paramaters=True,verbose=True)
        dbs.append(db)
        name = os.path.basename(fdb)
        name = name.split('.')[0]
        names.append(name)

    if mapping == 'tSNE':
        mol.plot_TSNE(dbs, names, output = out, random_max = random_max, delimiter = delimiter)
    if mapping == 'PCA':
        mol.plot_PCA(dbs, names, output = out, random_max = random_max, delimiter = delimiter)
