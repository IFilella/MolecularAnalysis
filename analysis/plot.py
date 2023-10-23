import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from MolecularAnalysis import mol
from MolecularAnalysis import moldb

def plotTrimap(dbs, names, output, random_max=None, delimiter=None, alg='Morgan4', colors=None,
               sizes=None, alphas=None, markers=None, figsize=(8,8), linewidths=None,
               n_iters=400, n_inliers=12, n_outliers=4, n_random=3, weight_temp=0.5, verbose=False):
    """
    Performe a Trimap with the fingerprint of the molecules of multiple MolDB
    an plot the two first components
    - dbs: list of MolDB
    - names: list of labels for the MolDB
    - output: output plot name
    - random_max: If an integer is given the PCA is going to be done
                  only for this number of molecules (default None)
    - delimiter: delimiter to slpit names
    - alg: Algorithm used to compute the Fingerprint (default Morgan4)
    - colors: list of matplotlib colors to color-code molecules from the
              different MolDBs
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
             different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
              of molecules from different MolDBs
    - linewidths: list of linewidths to adjust the linewidths of molecules from
                    different MolDBs
    - markers: list of markers to adjust the marker shape of molecules from
                  different MolDBs
    - figsize: output plot figure size (default (8,8))
    - n_iters: Number of iterations (default 400)
    - n_inliers: Number of nearest neighbors for forming the nearest neighbor triplets
                 (default 12)
    - n_outliers: Number of outliers for forming the nearest neighbor triplets (default 4)
    - n_random: Number of random triplets per point (default 3)
    - weight_temp: Temperature of the logarithm applied to the weights. Larger temperatures
                   generate more compact embeddings. weight_temp=0. corresponds to
                   no transformation (default 0.5)
    - verbose: If True get additional details (default False)
    """
    import trimap
    X, Y, S, A=_prepareReducer(dbs, names, random_max, delimiter, alg, sizes, alphas)
    if verbose: print('Computing trimap')
    tri=trimap.TRIMAP(n_iters=n_iters, distance='hamming', n_inliers=n_inliers,
                      n_outliers=n_outliers, n_random=n_random, weight_temp=weight_temp)
    trimap_results=tri.fit_transform(X)
    if verbose: print('Shape of trimap_results: ', trimap_results.shape)
    _plotReducer(reducer_results=trimap_results, Y=Y, output=output, colors=colors, sizes=S,
                 alphas=A, linewidths=linewidths, markers=markers, figsize=figsize)

def plotUMAP(dbs, names, output, random_max=None, delimiter=None, alg='Morgan4',
              colors=None, sizes=None, alphas=None, min_dist=0.1, n_neighbors=100,
              n_epochs=1000, markers=None, figsize=(8,8), linewidths=None, verbose=False):
    """
    Performe a UMAP with the fingerprint of the molecules of multiple MolDB
    an plot the two first components
    - dbs: list of MolDB
    - names: list of labels for the MolDB
    - output: output plot name
    - random_max: If an integer is given the PCA is going to be done
                  only for this number of molecules (default None)
    - delimiter: delimiter to slpit names
    - alg: Algorithm used to compute the Fingerprint (default Morgan4)
    - colors: list of matplotlib colors to color-code molecules from the
              different MolDBs
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
             different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
              of molecules from different MolDBs
    - min_dist: controls how tightly UMAP is allowed to pack points together
                (default 0.1)
    - n_neighbors: controls how UMAP balances local versus global structure in the
                   data (default 100)
    - n_epochs: Maximum number of iterations (default 1000)
    - linewidths: list of linewidths to adjust the linewidths of molecules from
                   different MolDBs
    - markers: list of markers to adjust the marker shape of molecules from
                 different MolDBs
    - figsize: output plot figure size (default (8,8))
    - verbose: If True get additional details (default False)
    """
    #from umap import UMAP
    import umap as mp
    X, Y, S, A=_prepareReducer(dbs, names, random_max, delimiter, alg, sizes, alphas)
    if verbose: print('Computing UMAP')
    umap=mp.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, metric='hamming')
    UMAP_results=umap.fit_transform(X)
    if verbose: print('Shape of UMAP_results: ', UMAP_results.shape)
    _plotReducer(reducer_results=UMAP_results, Y=Y, output=output, colors=colors, sizes=S,
                 alphas=A, linewidths=linewidths, markers=markers, figsize=figsize)

def plotTSNE(dbs, names, output, random_max=None, delimiter=None, alg='Morgan4',
              colors=None, sizes=None, alphas=None, linewidths=None, n_iter=1000,
              perplexity=30, early_exaggeration=12, learning_rate='auto', markers=None,
              figsize=(8,8)):
    """
    Performe a tSNE with the fingerprint of the molecules of multiple MolDB
    an plot the two first components
    - dbs: list of MolDB
    - names: list of labels for the MolDB
    - output: output plot name
    - random_max: If an integer is given the PCA is going to be done
                  only for this number of molecules (default None)
    - delimiter: delimiter to slpit names
    - alg: Algorithm used to compute the Fingerprint (default Morgan4)
    - colors: list of matplotlib colors to color-code molecules from the
              different MolDBs
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
             different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
              of molecules from different MolDBs
    - linewidths: list of linewidths to adjust the linewidths of molecules from
                   different MolDBs
    - n_iter: Maximum number of iterations (default 1500)
    - perplexity: paramater related to the number of nearest neighbors (default 30)
    - early_exaggeration: paramater that controls how tight natural clusters
                          in the original space are in the embedded space and how much space
                          will be between them (default 12)
    - markers: list of markers to adjust the marker shape of molecules from
                different MolDBs
    - figsize: output plot figure size (default (8,8))
    """
    from sklearn.manifold import TSNE
    X, Y, S, A=_prepareReducer(dbs, names, random_max, delimiter, alg, sizes, alphas)
    tsne=TSNE(n_components=2, verbose=1, learning_rate=learning_rate, init='pca',
              perplexity=perplexity, n_iter=n_iter, metric='hamming',
              early_exaggeration=early_exaggeration)
    tsne_results=tsne.fit_transform(X)
    _plotReducer(reducer_results=tsne_results, Y=Y, output=output, colors=colors,
                 sizes=S, alphas=A, linewidths=linewidths, markers=markers, figsize=figsize)

def plotPCA(dbs, names, output, random_max=None, delimiter=None, alg='Morgan4',
            colors=None, sizes=None, alphas=None, linewidths=None, markers=None,
            figsize=(8,8), verbose=False):
    """
    Performe a PCA with the fingerprint of the molecules of multiple MolDB
    an plot the two first components
    - dbs: list of MolDB
    - names: list of labels for the MolDB
    - output: output plot name
    - random_max: If an integer is given the PCA is going to be done
                  only for this number of molecules (default None)
    - delimiter: delimiter to slpit names
    - alg: Algorithm used to compute the Fingerprint (default Morgan4)
    - colors: list of matplotlib colors to color-code molecules from the
              different MolDBs
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
             different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
              of molecules from different MolDBs
    - linewidths: list of linewidths to adjust the linewidths of molecules from
                   different MolDBs
    - markers: list of markers to adjust the marker shape of molecules from
                different MolDBs
    - figsize: output plot figure size (default (8,8))
    - verbose: If True get additional details (default False)
    """
    from sklearn.decomposition import PCA
    X, Y, S, A=_prepareReducer(dbs, names, random_max, delimiter, alg,
                                sizes, alphas)
    if verbose: print('Computing PCA')
    pca=PCA(n_components=2)
    pca_results=pca.fit_transform(X)
    if verbose: print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    _plotReducer(reducer_results=pca_results, Y=Y, output=output, colors=colors,
                  sizes=S, alphas=A, linewidths=linewidths, markers=markers, figsize=figsize)

def _prepareReducer(dbs, names, random_max, delimiter, alg, sizes, alphas):
    """
    Get the X (fingerprints) data needed to run the dimensional reduction
    algorithm (PCA, UMAP, ...), and the data needed to plot it Y (MolDB labels),
    S (sizes), A (alpha transparencies)
    - dbs: list of MolDB
    - names: list of labels for the MolDB
    - random_max: If an integer is given the PCA is going to be done
                  only for this number of molecules (default None)
    - delimiter: delimiter to slpit names
    - alg: Algorithm used to compute the Fingerprint (default Morgan4)
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
             different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
              of molecules from different MolDBs
    """
    X, Y, S, A=[], [], [], []
    for i,db in enumerate(dbs):
        if delimiter==None:
            name=names[i]
        else:
            name=names[i].split(delimiter)[0]
            #name='_'.join(names[i].split(delimiter)[0:2])
        fps=db.getFingerprints(alg, random_max=random_max)
        fps=[fp for fp in fps if fp is not None]
        X.extend(fps)
        Y.extend([name]*len(fps))
        if sizes==None:
            S.extend([float(5)]*len(fps))
        else:
            S.extend([float(sizes[i])]*len(fps))
        if alphas==None:
            A.extend([float(0.8)]*len(fps))
        else:
            A.extend([float(alphas[i])]*len(fps))
    X=np.asarray(X)
    return X, Y, S, A

def _plotReducer(reducer_results, Y, output, colors, sizes, alphas, linewidths,
                  markers, figsize):
    """
    Plot the dimensional reduction results (PCA, UMAP, ...)
    - reducer_results: dimensional reduced data
    - Y: list of MolDBs labels for each molecule
    - output: output plot name
    - colors: list of matplotlib colors to color-code molecules from the
               different MolDBs
    - sizes: list of matplotlib sizes to adjust the sizes of molecules from
              different MolDBs
    - alphas: list of matplotlib transparency rates to asjust the transparency
               of molecules from different MolDBs
    - linewidths: list of linewidths to adjust the linewidths of molecules from
                  different MolDBs
    - markers: list of markers to adjust the marker shape of molecules from
               different MolDBs
    """
    df=pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1],
                         molDB=Y, sizes=sizes, alphas=alphas))
    plt.figure(figsize=figsize)
    if colors==None:
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis', hue='molDB', alpha=alphas,
                          size='sizes',linewidth=linewidths, style='molDB', markers=markers)
    else:
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis', hue='molDB', palette=colors,
                          alpha=alphas, size='sizes', linewidth=linewidths, style='molDB',
                          markers=markers)
    h,l=g.get_legend_handles_labels()
    n=len(set(df['molDB'].values.tolist()))
    #plt.legend(frameon=False, title=None)
    plt.legend(h[0:n+1],l[0:n+1]) #,bbox_to_anchor=(1.05, 1)), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output+'.pdf',dpi=500)
