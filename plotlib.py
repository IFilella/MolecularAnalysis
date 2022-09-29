#import mollib
from MolecularAnalysis import mollib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from umap import UMAP
import umap as mp
import trimap

def plot_trimap(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT', colors = None, sizes = None, alphas = None):
    X, Y, S, A = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg, sizes, alphas)
    print('Computing trimap')
    tri = trimap.TRIMAP()
    trimap_results = tri.fit_transform(X)
    print('Shape of trimap_results: ', trimap_results.shape)
    _plot_reducer(reducer_results = trimap_results, Y=Y, output=output, colors=colors, sizes=S, alphas=A)

def plot_UMAP(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT', colors = None, sizes = None, alphas = None, min_dist = 0.1, n_neighbors = 100, n_epochs = 1000):
    X, Y, S, A = _prepare_reducer(dbs, names, random_max, delimiter, fpsalg, sizes, alphas)
    print('Computing UMAP')
    umap = mp.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist,metric='hamming')
    UMAP_results = umap.fit_transform(X)
    print('Shape of UMAP_results: ', UMAP_results.shape)
    _plot_reducer(reducer_results = UMAP_results, Y=Y, output=output, colors=colors, sizes=S, alphas=A)

def plot_tSNE(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT',colors = None, sizes=None, alphas = None, n_iter=1000, perplexity=30):
    X, Y, S, A = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg, sizes, alphas)
    print('Computing TSNE')
    tsne = TSNE(n_components=2, verbose = 1, learning_rate='auto', init='pca', perplexity=perplexity, n_iter=n_iter, metric='hamming')
    tsne_results = tsne.fit_transform(X)
    _plot_reducer(reducer_results = tsne_results, Y=Y, output=output, colors=colors, sizes=S, alphas=A)

def plot_PCA(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg)
    print('Computing PCA')
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    _plot_reducer(reducer_results = pca_results, Y=Y, output=output)


def _prepare_reducer(dbs,names,random_max, delimiter, fpsalg, sizes,alphas):
    X = []
    Y = []
    S = []
    A = []
    for i,db in enumerate(dbs):
        if delimiter == None:
            name = names[i]
        else:
            name = names[i].split(delimiter)[0]
            #name = '_'.join(names[i].split(delimiter)[0:2])
        print(name)
        fps = db.get_fingerprints(fpsalg, random_max)
        X.extend(fps)
        Y.extend([name]*len(fps))
        if sizes == None:
            S.extend([float(5)]*len(fps))
        else:
            S.extend([float(sizes[i])]*len(fps))
        if alphas == None:
            A.extend([float(0.8)]*len(fps))
        else:
            A.extend([float(alphas[i])]*len(fps))
    X = np.asarray(X)
    return X, Y, S, A

def _plot_reducer(reducer_results,Y,output,colors,sizes,alphas):
    df = pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1],  molDB = Y, sizes=sizes, alphas=alphas))
    plt.figure()
    if colors == None:
        g = sns.scatterplot('xaxis', 'yaxis', data=df, hue='molDB', alpha=alphas, size='sizes')#,style='molDB')
    else:
        g = sns.scatterplot('xaxis', 'yaxis', data=df, hue='molDB', palette=colors ,alpha=alphas, size='sizes')
    h,l = g.get_legend_handles_labels()
    n = len(set(df['molDB'].values.tolist()))
    plt.legend(h[0:n+1],l[0:n+1])#,bbox_to_anchor=(1.05, 1)), loc=2, borderaxespad=0.)
    plt.tight_layout()
    if output != None:
        plt.savefig(output+'.pdf',dpi=300)
