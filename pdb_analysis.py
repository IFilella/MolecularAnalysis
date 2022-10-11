import glob
from prody import *
import numpy as np
import os
import time
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster 
from scipy.spatial.distance import pdist
import pickle
import re
import shutil
import distinctipy

def get_protChains(pdbs,outname,delimiter=None,upresfilter=None,lowresfilter=None,verbose=True):
    """
    From a pdbs list retrieve all its chains (filtered if asked) into individual pdbs with only protein
    pdbs: 'list'. PDBs list
    outname: 'str'. Prefix to add to the new PDB chains
    delimiter: 'str'. Delimiter to obtain an identifier from each PDB name
    upresfilter: 'int'. Residue upper threhsold. Maximum number of residues allowed per chain
    lowresfilter: 'int'. Residue lower threshold. Minimum number of residues per chain
    """
    PDBnames = [os.path.basename(pdb) for pdb in pdbs]
    if delimiter != None:
        IDs = [ pdbname.replace(".pdb","").split(delimiter)[0] for pdbname in PDBnames]
    else:
        IDs = [ pdbname.replace(".pdb","") for pdbname in PDBnames]
    for i,pdb in enumerate(pdbs):
        pdbname = PDBnames[i]
        if verbose: print(pdbname)
        structurePDB = parsePDB('%s'%(pdb)) #From PDB to prody atom class
        hvPDB = structurePDB.getHierView() #Get prody hierarchical view
        chains = [chain.getChid() for chain in hvPDB] #Get chain identifiers
        for chain in chains:
            if verbose: print(chain)
            nresidues =  hvPDB[chain].numResidues()
            if upresfilter != None:
                if nresidues > upresfilter:
                    print("Too many residues for chain %s in %s"%(chain,pdbname))
                    continue
            if lowresfilter != None:
                if nresidues < lowresfilter:
                    print("Too few residues for chain %s in %s"%(chain,pdbname))
                    continue
            writePDB('%s_%s%s.pdb'%(outname,IDs[i],chain),hvPDB[chain].select('protein')) #Save only protein

def pdb_extract(pdb_dir,schrodinger_path):
    """This function extracts the target structure (receptor), the ligands,
    the waters and the ions/cofactors of the list of pdb files."""

    os.chdir(pdb_dir)
    PDBs = glob.glob("*.pdb*")
    for PDB in PDBs:
        if '.pdb.gz' in PDB:
            cmd1 = "gunzip %s"%PDB
            os.system(cmd1)
            PDB = PDB.replace('.pdb.gz','.pdb')
            ID = PDB.replace(".pdb","")
        elif '.pdb' in PDB:
            ID = PDB.replace(".pdb","")
        print('Extracting %s ligands...' %(ID))
        cmd2 = '%srun split_structure.py -m pdb %s %s.pdb -many_files'%(schrodinger_path,PDB,ID)
        os.system(cmd2)
    _organize_extraction_files()

def _organize_extraction_files():
    """This function organize all the output files obtained in the pdb extraction
    into their respective directories."""

    os.mkdir("ligand")
    os.mkdir("receptor")
    os.mkdir("water")
    os.mkdir("cof_ion")
    
    currentdir = os.getcwd()
    files = glob.glob("%s/*pdb*"%currentdir)

    for filename in files:
        if re.search("ligand", filename):
            shutil.move("%s"%(filename), "ligand/%s"%(filename))
        if re.search("receptor", filename):
            shutil.move("%s"%(filename), "receptor/%s"%(filename))
        if re.search("water", filename):
            shutil.move("%s"%(filename), "water/%s"%(filename))
        if re.search("cof_ion", filename):
            shutil.move("%s"%(filename), "cof_ion/%s"%(filename))

def pdb_superimpose(mob_pdb,fix_pdb,outdir,verbose=True):
    structureFIX = parsePDB(fix_pdb)
    structureMOB = parsePDB(mob_pdb)
    nameMOB = os.path.basename(mob_pdb).replace(".pdb","")
    try:
        matchAlign(structureMOB,structureFIX)
        writePDB('%s/%s_super.pdb'%(outdir,nameMOB),structureMOB)
        return 0
    except:
        print('Superimposition between %s and %s failed'%(mob_pdb,fix_pdb))
        return -1

def pdbs_superimposition(pdbs,fix_pdb,outdir,verbose=True):
    """
    Given a list of mobile pdbs and a fix superimpose all mobile elements to the fix pdb
    pdbs: 'list'. PDBs list of mobile elements
    fix_pdb: 'str'. Single PDB which will be fixed during the multiple superimpositions (fix element)
    outdir: 'str'. Directory to store all the superimposed PDBs
    """ 
    errors = 0
    name_fix = os.path.basename(fix_pdb)
    for pdb in pdbs:
        name_mobile = os.path.basename(pdb)
        print('Superimposition between fix PDB %s and mobile PDB %s'%(name_fix,name_mobile))
        aux_error = pdb_superimpose(pdb,fix_pdb,outdir,verbose)
        errors += aux_error
    print('%d couldn\'t be superimpose'%-errors)
    
    #for i, pdb in enumerate(pdbs):
    #    #Superimpose using TMalign
    #    pdbname = PDBnames[i]
    #    if verbose: print(pdbname)
    #    outname = '%s/%s'%(outdir,IDs[i])
    #    outname = outname.replace('//','/')
    #    TMalign_cmd = 'TMalign %s %s -o %s'%(pdb, fix_pdb,outname)
    #    print(TMalign_cmd)
    #    os.system(TMalign_cmd)
    #    os.system('rm %s %s_all %s_all_atm_lig %s_atm'%(outname,outname,outname,outname))
    #    os.system('mv %s_all_atm %s_all_atm.pdb'%(outname,outname))
    #    
    #    #Extract mobile pdb from the TMalign output
    #    structurePDB = parsePDB('%s_all_atm.pdb'%(outname))
    #    #Load Prody Hierarchical Views
    #    hvPDB = structurePDB.getHierView()
    #    #Write superimpose mobile element as a pdb
    #    aux = '%s/%s_super.pdb'%(outdir,IDs[i])
    #    aux = aux.replace("//","/")
    #    writePDB(aux,hvPDB['A'])
    #    os.system('rm %s_all_atm.pdb'%(outname))

def prepWizard(pdbs,schroodinger_path,delimiter=None,outfmt='mae',max_processes=4):
    """
    Given a list of pdbs, use PrepWizard from the Schroodinger premium package to preapre the PDB(protonate, add missing side chains etc...)
    pdbs: 'list'. PDB list of elements to prepare
    schroodinger_path: 'str'. Global path of Schroodinger 
    delimiter: 'str'. Delimiter to obtain an identifier from each PDB name
    outfmt: 'str'. Outfile format. Either .mae or .pdb
    max_processes: Number of processors used to paralalize the different executions
    """
    if outfmt != 'mae' and outfmt != 'pdb':
        raise ValueError('outfmt must be either mae or pdb')
    PDBnames = [os.path.basename(pdb) for pdb in pdbs]
    if delimiter != None:
        IDs = [ pdbname.replace(".pdb","").split(delimiter)[0] for pdbname in PDBnames]
    else:
        IDs = [ pdbname.replace(".pdb","") for pdbname in PDBnames]
    
    cmd_prepW = ['%s/utilities/prepwizard -fillsidechains -WAIT %s %s_prep.%s'%(schroodinger_path,pdb,IDs[i],outfmt) for i,pdb in enumerate(pdbs)]
    cmd_prepW = [cmd.replace('//','/') for cmd in cmd_prepW]
    processes = set()

    for cmd in cmd_prepW:
        print(cmd)
        processes.add(subprocess.Popen(cmd,shell=True))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

def _clean_prepWizard(outdir,outfmt='mae'):
    """
    Move the PrepWizard output to an specified directory
    outdir: 'str'. Output directory
    outfmt: 'str'. Outfile format of the PrepWizard. Either .mae or .pdb
    """
    if outfmt != 'mae' and outfmt != 'pdb':
        raise ValueError('outfmt must be either mae or pdb')
    os.system('mv *_prep.%s %s'%(outfmt,outdir))
    logdir = '%s/logs'%outdir
    logdir = logdir.replace('//','/')
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    os.system('mv *.mae %s'%logdir)
    os.system('mv *.log %s'%logdir)

def siteMap(maes,asl,schroodinger_path,delimiter=None,outfmt='mae',max_processes=4):
    """
    Run a SiteMap calculation for a list of MAEs (can't be pdbs).
    maes: 'list'. MAE list of elements
    asl: 'str'. ASL (atom specification Language)
    schroodinger_path: 'str'. Global path of Schroodinger
    delimiter: 'str'. Delimiter to obtain an identifier from each MAE name
    outfmt: 'str'. Outfile format. Either .mae or .pdb
    max_processes: Number of processors used to paralalize the different executions
    """
    MAEnames = [os.path.basename(mae) for mae in maes]
    if delimiter != None:
        IDs = [ maename.replace(".mae","").split(delimiter)[0] for maename in MAEnames]
    else:
        IDs = [ maename.replace(".mae","") for maename in MAEnames]
    
    cmd_SiteMap = ['%s/sitemap -j %s -prot %s -sitebox 12 -resolution standard -reportsize 100 -writestructs no -maxsites 1 -siteasl "%s" -WAIT'%(schroodinger_path,IDs[i],mae,asl) for i,mae in enumerate(maes)]
    cmd_SiteMap = [cmd.replace("//","/") for cmd in cmd_SiteMap]

    processes = set()

    for cmd in cmd_SiteMap:
        print(cmd)
        processes.add(subprocess.Popen(cmd,shell=True))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

def _clean_siteMap(outdir,outfmt='maegz'):
    """
    Move the SiteMap output to an specified directory
    outdir: 'str'. Output directory
    outfmt: 'str'. Outfile format of the PrepWizard. Either .mae or .pdb
    """
    if outfmt != 'maegz':
        raise ValueError('outfmt must be maegz')
    os.system('mv *.%s %s'%(outfmt,outdir))
    logdir = '%s/logs'%outdir
    logdir = logdir.replace('//','/')
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    os.system('mv *.vis %s'%logdir)
    os.system('mv *.smap %s'%logdir)
    os.system('mv *.log %s'%logdir)

def _group_siteMap(sites,out,outdir,schroodinger_path):
    """
    Group all volume sites from SiteMap into a single mae file
    sites:
    out:
    schroodinger_path: 'str'. Global path of Schroodinger
    """
    conc_sites = ''
    for site in sites:
        conc_sites = conc_sites + ' ' + site

    cmd = '%s/utilities/structcat -imae %s -omae %s'%(schroodinger_path,conc_sites,out)
    cmd = cmd.replace('//','/')
    print(cmd)
    os.system(cmd)
    try:
        aux = 'mv %s %s'%(out,outdir)
        os.system(aux)
    except:
        print('wrong outdir')

def get_volumeOverlapMatrix(sites,out,schroodinger_path,max_processes=4):
    """
    Generate pairwise volume overlap matrix
    sites: 'str'. single file containing multiple SitMap files
    """
    cmd = '%s/run volume_cluster.py -j %s -HOST localhost:%d -sc -r 2 %s'%(schroodinger_path,out,max_processes,sites)
    cmd = cmd.replace('//','/')
    print(cmd)
    os.system(cmd)

def _clean_volumeMatrix(out,outdir):
    """
    Move the VolumeMatrix output to an specified directory
    out: 'str'
    outdir: 'str'. Output directory
    """
    os.system('mv *.csv %s'%(outdir))
    logdir = '%s/logs/'%outdir
    logdir = logdir.replace('//','/')
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    os.system('mv *.mae %s'%logdir)
    os.system('mv *.log %s'%logdir)

def _uncompress_maegz(inp,schroodinger_path):
    """
    inp:
    schroodinger_path:
    """
    out = inp.replace('.maegz','.mae')
    cmd = '%s/utilities/structcat -imae %s -omae %s'%(schroodinger_path,inp,out)
    cmd = cmd.replace('//','/')
    print(cmd)
    os.system(cmd)


class VolumeOverlappingMatrix(object):
    """""
    """""
    def __init__(self, csv, IDs=None, IDs_shroodinger=None, identifier = None, delimiter = '', del_position=0):
        """
        csv: 'str'. csv file containing the volume overlapping matrix
        IDs: 'list'. list of IDs to replace csv indices 
        IDs_shroodinger: 'str'. Input file used to compute the volume overlpaping matrix with SiteMap
        """
        self.matrix = pd.read_csv(csv,delimiter=',',index_col=0)
        if IDs != None and IDs_shroodinger == None:
            if len(IDs) != self.matrix.shape[0]:
                raise ValueError('IDs should have the same length as rows/cols of csv file')
            self.IDs = IDs
            self.matrix.set_axis(self.IDs, axis=1, inplace=True)
            self.matrix.set_axis(self.IDs, axis=0, inplace=True)
        elif IDs == None and IDs_shroodinger != None:
            if identifier == None:
                raise ValueError('To select the IDs from IDs_shroodinger an identifier is needed')
            self._get_IDs_shroodinger(IDs_shroodinger,identifier,delimiter,del_position) 
        if IDs != None and IDs_shroodinger != None:
            raise ValueError('Pass either an IDs list or a IDs_shroodinger file') 

    def _get_IDs_shroodinger(self, vm_input, identifier, delimiter='',del_position=0):
        """
        Get the IDs needed for the volume overlapping matrix from the input file used to compute it.
        vm_input: 'str'. Input file used to compute the volume overlapping matrix (mae)
        identifier: 'str'.
        """
        f = open(vm_input,'r')
        IDs = []
        for line in f:
            if identifier in line:
                ID = line.split(delimiter)[del_position]
                if ID not in IDs:
                    IDs.append(ID)
        self.matrix.set_axis(IDs, axis=1, inplace=True)
        self.matrix.set_axis(IDs, axis=0, inplace=True)
    
    def plot_hierarchical(self,out,fontsize=1):
        """
        """
        sns.set(font_scale=fontsize)
        cg = sns.clustermap(self.matrix,cmap="RdBu_r",yticklabels=True,xticklabels=True,vmin=0,vmax=1)
        plt.savefig(out,dpi=300)

    def plot_hierarchical_labeled(self, properties_df, features, out, fontsize = 1, printlabels = False, ucolors = None):
        """
        Hierarchical clustermap with color and row coloring according to a given feature of
        properties_df. 'pandas DataFrame'
        features. 'list'. Columns/properties to be used during the coloring
        out. 'str'. Outname
        fontsize. 'int'. Fontsize
        ucolors. 'list'.
        """
        Ncolors = 0
        for feature in features:
            Ncolors += len(properties_df[feature].unique())
        if ucolors == None:
            rgb_colors = distinctipy.get_colors(Ncolors)
        else:
            if len(ucolors) != Ncolors: raise ValueError('Incorrect number of colors')
            else:
                rgb_colors = ucolors
        columns = self.matrix.columns.tolist()
        colorindex = 0
        list_dfcolors = []
        list_luts = []
        for i,feature in enumerate(features):
            colors = []
            nflavours = len(properties_df[feature].unique())
            lut = dict(zip(properties_df[feature].unique(),rgb_colors[colorindex:colorindex+nflavours]))
            list_luts.append(lut)
            colorindex += nflavours
            print(lut)
            for j,column in enumerate(columns):
                ptype = properties_df.loc[column,feature]
                colors.append(lut[ptype])
            _dfcolors = pd.DataFrame({feature: colors}, index=columns)
            list_dfcolors.append(_dfcolors)
        dfcolors = pd.concat(list_dfcolors,axis=1)

        sns.set(font_scale = fontsize)
        cg = sns.clustermap(self.matrix,cmap="RdBu_r", row_colors=dfcolors, col_colors=dfcolors, yticklabels=printlabels,xticklabels=printlabels,vmin=0,vmax=1)
        
        #Plot legend
        for i,feature in enumerate(features):
            for flavour in properties_df[feature].unique():
                cg.ax_col_dendrogram.bar(0, 0, color=list_luts[i][flavour], label=flavour, linewidth=0)
                cg.ax_col_dendrogram.legend(loc='lower left', bbox_to_anchor=(0.9, 0.5) ,ncol=1)

        plt.savefig(out+'.pdf',dpi=300)
         

    def get_dendrogram(self,verbose=True):
        """
        """
        dendrogram = linkage(self.matrix, 'average',metric='euclidean')
        self.dendrogram = dendrogram
        c, coph_dists = cophenet(self.dendrogram, pdist(self.matrix))
        self.c = c
        self.coph_dists = coph_dists
        if verbose:
            print('Cophenetic Correlation Coefficient: ', str(c))

    def fancy_dendrogram(self,*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    def plot_dendrogram(self,out,max_d,annotate_above,p=None):
        """
        """
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index',fontsize=16)
        plt.ylabel('distance',fontsize=16)
        if p==None:
            self.fancy_dendrogram(self.dendrogram,leaf_rotation=90., leaf_font_size=8.,labels=self.IDs,max_d=max_d,annotate_above=annotate_above)
        else:
            self.fancy_dendrogram(self.dendrogram,leaf_rotation=90., leaf_font_size=8.,labels=self.IDs,max_d=max_d,annotate_above=annotate_above,p=p,truncate_mode='lastp',show_contracted=True)
        ax = plt.gca()
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='major', labelsize=14)
        plt.savefig(out,dpi=300)

    def get_dendrogram_clusters(self,max_d):
        clusters = fcluster(self.dendrogram, max_d, criterion='distance')
        self.clusters = clusters
        clusters_dic = {}
        for i,id in enumerate(self.IDs):
            cluster = self.clusters[i]
            if cluster not in clusters_dic:
                clusters_dic[cluster]=[id]
            else:
                clusters_dic[cluster].append(id)
        self.clusters_dic = clusters_dic

    def save_dendrogram_clusters(self,out,verbose=True): 
        if verbose:
            print(self.clusters_dic)
            for cluster in self.clusters_dic.keys():
                print(cluster,len(self.clusters_dic[cluster]))
        
        with open(out+'.p', 'wb') as handle:
            pickle.dump(self.clusters_dic, handle)

    def get_dendrogram_clustersCenters(self):
        clustersCenters = {}
        for cluster in self.clusters_dic.keys():
            cluster_elements = self.clusters_dic[cluster]
            volume_matrix_cluster = self.matrix.loc[cluster_elements,cluster_elements]
            volume_matrix_cluster['mean'] = volume_matrix_cluster.mean(axis=1)
            center = volume_matrix_cluster['mean'].idxmax() 
            print('center cluster%d: '%cluster, center)
            clustersCenters[cluster] = center
        self.clustersCenters = clustersCenters 
    
    def remove_cluster_outliers(self, list_clusters):
        outlayer_clusters = list_clusters

        for cluster in self.clusters_dic.keys():
            if cluster in outlayer_clusters:
                #print(cluster)
                outlayers = self.clusters_dic[cluster]
                #print(outlayers)
                self.matrix.drop(labels=outlayers,axis=1,inplace=True)
                self.matrix.drop(labels=outlayers,axis=0,inplace=True)
                for outlayer in outlayers:
                    self.IDs.remove(outlayer)

    def organize_cluster_structures(self, results_dir, struct_dir, site_dir):
        if not os.path.isdir('%s/clusters'%results_dir):
            os.system('mkdir %s/clusters'%results_dir)

        for cluster in self.clusters_dic.keys():
            cluster_elements = self.clusters_dic[cluster]
            for struct in cluster_elements:
                element_site = '%s/Mpro_%s_super_out.maegz'%(site_dir, struct)
                element_prep = "%s/Mpro_%s_super_prep.mae"%(struct_dir, struct)
                if not os.path.isdir('%s/clusters/%d'%(results_dir, cluster)):
                    os.system('mkdir %s/clusters/%d'%(results_dir, cluster))
                cmd1 = 'cp %s %s/clusters/%d'%(element_site, results_dir, cluster)
                cmd2 = 'cp %s %s/clusters/%d'%(element_prep, results_dir, cluster)
                os.system(cmd1)
                os.system(cmd2)

    def get_closest_elements_cluster_center(self, num_elements):
        for cluster in self.clusters_dic.keys():
            print('Cluster %s:' %cluster)
            center = self.clustersCenters[cluster]
            elements = self.clusters_dic[cluster]
            submatrix = self.matrix.loc[elements,elements]
            print(submatrix.nlargest(num_elements, [center])[center])

if __name__ == '__main__':
    schroodinger_path = '/data/general_software/schrodinger2019-1'

    #1: Get all target chains (filtered if asked) into individual pdbs
    #TARGs = glob.glob('tests/PDBs/*receptor*')
    #get_protChains(pdbs=TARGs,outname='tests/PDBs/CDK2',delimiter='_',upresfilter=300,lowresfilter=280)
    
    #2: Superimpose all targets to a fix target
    #TARGs = glob.glob('tests/PDBs/CDK2*')
    #pdbs_superimposition(pdbs=TARGs,fix_pdb='tests/PDBs/CDK2_1b38A.pdb',outdir='tests/superimpositions/')
    
    #3: Prepare the targets with PrepWizard from Schroodinger premium package 
    #TARGs = glob.glob('tests/superimpositions/CDK2*')
    #prepWizard(pdbs=TARGs,delimiter='_super',schroodinger_path=schroodinger_path,outfmt='pdb',max_processes=30)
    #_clean_prepWizard(outdir='tests/prepWizard',outfmt='pdb')
    
    #4: Compute the volume of each target specific binding site (single one sourranding LYS33 NZ)
    #TARGs = glob.glob('tests/prepWizard/*_prep.mae') 
    #siteMap(maes=TARGs,asl = "(res.num 33) AND ((atom.ptype \' NZ \'))", schroodinger_path=schroodinger_path,delimiter='_prep',outfmt='mae',max_processes=30)
    #_clean_siteMap(outdir='tests/siteMap')
    
    #5: Find which targets do not have a binding site arround the specified atom
    #TARGs = glob.glob('tests/prepWizard/*_prep.mae')
    #sites = glob.glob('tests/siteMap/*_out.maegz')
    #TARGs_IDs = [os.path.basename(TARG).split('_')[1] for TARG in TARGs]
    #sites_IDs = [os.path.basename(site).split('_')[1] for site in sites]
    #print(set(TARGs_IDs)-set(sites_IDs))
    
    #6: Group all SiteMap sites into a single file
    #_group_siteMap(sites=sites,out='CDK2_sites.maegz',outdir='tests/siteMap/',schroodinger_path=schroodinger_path)
    #_uncompress_maegz(inp='tests/siteMap/CDK2_sites.maegz',schroodinger_path=schroodinger_path)
    
    #7: Get the volume overlapping matrix of the target sites
    #get_volumeOverlapMatrix(sites='tests/siteMap/CDK2_sites.maegz',out='CDK2_volumeMatrix_r2',schroodinger_path=schroodinger_path,max_processes=4)
    #_clean_volumeMatrix(out='CDK2_volumeMatrix_r2',outdir='tests/volumeMatrix/')
    
    #8: Analyse the volume overlapping matrix
    #f = open('tests/siteMap/CDK2_sites.mae','r')
    #IDs = []
    #for line in f:
    #    if 'CDK2' in line:
    #        ID = line.split('_')[1]
    #        if ID not in IDs:
    #            IDs.append(ID)
    #volume_matrix = VolumeOverlappingMatrix('tests/volumeMatrix/CDK2_volumeMatrix_r2.csv',IDs)
    #print(volume_matrix.matrix)
    #volume_matrix.plot_hierarchical(out='tests/volumeMatrix/CDK2_volumematrix_r2.pdf')
    #volume_matrix.get_dendrogram()
    #volume_matrix.plot_dendrogram(out='tests/volumeMatrix/CDK2_dendrogram_r2.pdf',max_d=5,annotate_above=20)
    #volume_matrix.plot_dendrogram(out='tests/volumeMatrix/CDK2_dendrogram2_r2.pdf',max_d=0.6,annotate_above=20)
    #volume_matrix.plot_dendrogram(out='tests/volumeMatrix/CDK2_dendrogram2_r3.pdf',max_d=1.95,annotate_above=0.6,p=3)
    #volume_matrix.get_dendrogram_clusters(max_d=0.6)
    #print(volume_matrix.clusters)
    #print(len(volume_matrix.clusters))
    #print(set(volume_matrix.clusters))
    #volume_matrix.save_dendrogram_clusters(out='tests/volumeMatrix/CDK2_dendrogram_clusters')
    #
    #with open('tests/volumeMatrix/CDK2_dendrogram_clusters.p', 'rb') as handle:
    #    clusters_dic = pickle.load(handle)

    #volume_matrix.get_dendrogram_clustersCenters()
