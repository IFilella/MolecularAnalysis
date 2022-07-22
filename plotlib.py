import mollib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import trimap

def plot_trimap(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg)
    print('Computing trimap')
    tri = trimap.TRIMAP()
    trimap_results = tri.fit_transform(X)
    print('Shape of trimap_results: ', trimap_results.shape)
    _plot_reducer(reducer_results = trimap_results, Y=Y, output=output)

def plot_UMAP(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg)
    print('Computing UMAP')
    umap = UMAP(n_neighbors=100, n_epochs=1000)
    UMAP_results = umap.fit_transform(X)
    print('Shape of UMAP_results: ', UMAP_results.shape)
    _plot_reducer(reducer_results = UMAP_results, Y=Y, output=output)

def plot_tSNE(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg)
    print('Computing TSNE')
    tsne = TSNE(n_components=2, verbose = 1, learning_rate='auto',init='pca')
    tsne_results = tsne.fit_transform(X)
    _plot_reducer(reducer_results = tsne_results, Y=Y, output=output)

def plot_PCA(dbs,names,output=None, random_max = None, delimiter = None, fpsalg = 'RDKIT'):
    X, Y = _prepare_reducer(dbs,names,random_max, delimiter, fpsalg)
    print('Computing PCA')
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    _plot_reducer(reducer_results = pca_results, Y=Y, output=output)


def _prepare_reducer(dbs,names,random_max, delimiter, fpsalg):
    X = []
    Y = []
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
    X = np.asarray(X)
    return X, Y

def _plot_reducer(reducer_results,Y,output):
    df = pd.DataFrame(dict(xaxis=reducer_results[:,0], yaxis=reducer_results[:,1],  molDB = Y))
    plt.figure()
    sns.scatterplot('xaxis', 'yaxis', data=df, hue='molDB',alpha = 0.8, s=5,style='molDB')
    if output != None:
        plt.savefig(output+'.png',dpi=300)
