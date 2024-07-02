from MolecularAnalysis.utils import protstruct
import glob
import os
import pickle

schroodinger_path='/home/cactus/Programs/schrodinger2023-1/'

if not os.path.exists('testout'): os.system('mkdir testout')

#0:
if not os.path.exists('testout/PDBs'): os.system('mkdir testout/PDBs')
protstruct.splitPDBs(pdb_dir='../data/PDBs', out_dir='testout/PDBs',
              schrodinger_path=schroodinger_path)

#1: Get all target chains (filtered if asked) into individual pdbs
TARGs=glob.glob('testout/PDBs/receptor/*')
if not os.path.exists('testout/PDBs/chains'): os.system('mkdir testout/PDBs/chains')
protstruct.getChains(pdbs=TARGs, out_dir='testout/PDBs/chains/', delimiter='_',
              upresfilter=300, lowresfilter=280)

#2: Superimpose all targets to a fix target
TARGs=glob.glob('testout/PDBs/chains/*')
if not os.path.exists('testout/PDBs/superimpositions'): os.system('mkdir testout/PDBs/superimpositions')
protstruct.superimposePDBs(pdbs=TARGs, fix_pdb=TARGs[0], out_dir='testout/PDBs/superimpositions/')

#3: Prepare the targets with PrepWizard from Schroodinger premium package
TARGs=glob.glob('testout/PDBs/superimpositions/*')
protstruct.runPrepWizard(pdbs=TARGs, delimiter='_super', schroodinger_path=schroodinger_path,
              out_fmt='mae', max_processes=30)
if not os.path.exists('testout/PDBs/prepWizard'): os.system('mkdir testout/PDBs/prepWizard')
protstruct.cleanPrepWizard(out_dir='testout/PDBs/prepWizard',out_fmt='mae')

#4: Compute the volume of each target specific binding site (single one sourranding LYS33 NZ)
TARGs=glob.glob('testout/PDBs/prepWizard/*_prep.mae')
print(TARGs)
protstruct.runSiteMap(maes=TARGs,asl="(res.num 33) AND ((atom.ptype \' NZ \'))",
               schroodinger_path=schroodinger_path, delimiter='_prep', max_processes=30)
if not os.path.exists('testout/PDBs/siteMap'): os.system('mkdir testout/PDBs/siteMap')
protstruct._cleanSiteMap(out_dir='testout/PDBs/siteMap')

#5: Find which targets do not have a binding site arround the specified atom
TARGs=glob.glob('testout/PDBs/prepWizard/*_prep.mae')
sites=glob.glob('testout/PDBs/siteMap/*_out.maegz')
TARGs_IDs=[os.path.basename(TARG).split('_')[1] for TARG in TARGs]
sites_IDs=[os.path.basename(site).split('_')[1] for site in sites]
print(set(TARGs_IDs)-set(sites_IDs))

#6: Group all SiteMap sites into a single file
protstruct._groupSiteMap(sites=sites, out_name='siteMaps.maegz', out_dir='testout/PDBs/siteMap/',
                  schroodinger_path=schroodinger_path)
protstruct._uncompressMaegz(maegz='testout/PDBs/siteMap/siteMaps.maegz',
                      schroodinger_path=schroodinger_path)
#7: Get the volume overlapping matrix of the target sites
protstruct.getVolumeOverlapMatrix(sites='testout/PDBs/siteMap/siteMaps.maegz',
                           out='volumeOverlappingMatrix',schroodinger_path=schroodinger_path,
                           max_processes=4)
if not os.path.exists('testout/PDBs/volumeMatrix'): os.system('mkdir testout/PDBs/volumeMatrix')
protstruct._cleanVolumeOverlapMatrix(matrix='volumeOverlappingMatrix.csv',
                          out_dir='testout/PDBs/volumeMatrix/')

#8: Analyse the volume overlapping matrix
f=open('testout/PDBs/siteMap/siteMaps.mae','r')
IDs=[]
for line in f:
    if '_site_' in line:
        ID=line.split('_site_')[0].replace(' ','')
        if ID not in IDs:
            IDs.append(ID)
IDs=list(set(IDs))
volume_matrix=protstruct.VolumeOverlappingMatrix('testout/PDBs/volumeMatrix/volumeOverlappingMatrix.csv',IDs)
print(volume_matrix.matrix)
volume_matrix.plot_hierarchical(out='testout/PDBs/volumeMatrix/volumeOverlappingMatrix.png')
volume_matrix.get_dendrogram()
volume_matrix.plot_dendrogram(out='testout/PDBs/volumeMatrix/dendrogram.pdf', max_d=5,
                              annotate_above=20)
volume_matrix.plot_dendrogram(out='testout/PDBs/volumeMatrix/dendrogram2.pdf', max_d=0.6,
                              annotate_above=20)
volume_matrix.plot_dendrogram(out='testout/PDBs/volumeMatrix/dendrogram3.pdf', max_d=1.95,
                              annotate_above=0.6, p=3)
volume_matrix.get_dendrogram_clusters(max_d=0.6)
print(volume_matrix.clusters)
print(len(volume_matrix.clusters))
print(set(volume_matrix.clusters))
volume_matrix.save_dendrogram_clusters(out='testout/PDBs/volumeMatrix/dendrograms_clusters')

with open('testout/PDBs/volumeMatrix/dendrograms_clusters.p', 'rb') as handle:
    clusters_dic=pickle.load(handle)

volume_matrix.get_dendrogram_clustersCenters()
