import argparse
import os
import sys
sys.path.insert(1, '../')
import mollib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Given a list of fragment databases with the format (SMILE simSMILES IDs), plot them into a mapping such as t-SNE, PCA, UMAP or trimap')
    parser.add_argument('-d', dest="dbs", nargs = '+', help = "List of databases of fragments",required=True)
    parser.add_argument('-m', dest="mapping", default = 'tSNE', help = "Mapping method (tSNE, PCA, UMAP or trimap)",required=True)
    parser.add_argument('-o', dest="out", default = None, help = "Map output name")    
    parser.add_argument('-r', dest= "random_max", default = 5000, help = "Randomly select X elements of each databse")
    parser.add_argument('--txt',default=False,action='store_true',help='To pass either a txt moldb (if True) or a pickle moldb (if False). False by default')
    parser.add_argument('--del', dest="delt", default = None, help = "Delimiter for the database name")
    parser.add_argument('--fps', dest="fpsalg", default = 'RDKIT', help = "Algorthim to generate the molecular fingerprints. Must be RDKIT, Morgan2, Morgan4 or Morgan8")
    args = parser.parse_args()
    
    fdbs = args.dbs
    mapping = args.mapping
    if mapping not in ['tSNE','UMAP','PCA','trimap']:
        raise ValueError('%s is not a valid mapping method. Mapping must be either t-SNE, PCA, UMAP or trimap.'%mapping)
    out = args.out
    txt = args.txt
    delimiter = args.delt
    random_max = int(args.random_max)
    fpsalg = args.fpsalg
    if fpsalg not in ['RDKIT', 'Morgan2', 'Morgan4', 'Morgan8']:
        raise ValueError('%s is not a valid FingerPrint method. It must be RDKIT, Morgan2, Morgan4 or Morgan8.'%mapping)

    dbs = []
    names = []
    for fdb in fdbs:
        if txt:
            db = mollib.MolDB(txtDB=fdb,paramaters=True,verbose=True)
            db.save_MolDB(fdb.replace('.txt',''))
        else:
            db = mollib.MolDB(dicDB=fdb,paramaters=True,verbose=True)
        dbs.append(db)
        name = os.path.basename(fdb)
        name = name.split('.')[0]
        names.append(name)

    if mapping == 'tSNE':
        mollib.plot_TSNE(dbs, names, output = out, random_max = random_max, delimiter = delimiter, fpsalg = fpsalg)
    if mapping == 'PCA':
        mollib.plot_PCA(dbs, names, output = out, random_max = random_max, delimiter = delimiter, fpsalg = fpsalg)
    if mapping == 'UMAP':
        mollib.plot_UMAP(dbs, names, output = out, random_max = random_max, delimiter = delimiter, fpsalg = fpsalg)
    if mapping == 'trimap':
        mollib.plot_trimap(dbs, names, output = out, random_max = random_max, delimiter = delimiter, fpsalg = fpsalg)
