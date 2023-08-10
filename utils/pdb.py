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

def checkPath(path):
    """
    Ensure correct path
    - path: path to correct
    """
    if path[-1]!='/': path+='/'
    return path

def getChains(pdbs, out_dir, delimiter=None, upresfilter=None, lowresfilter=None, verbose=False):
    """
    From a pdbs list retrieve all its chains (filtered if asked) into individual pdbs
    with only protein
    - pdbs: 'list'. PDBs list
    - out_dir: 'str'. Prefix to add to the new PDB chains
    - delimiter: 'str'. Delimiter to obtain an identifier from each PDB name
    - upresfilter: 'int'. Residue upper threhsold. Maximum number of residues allowed per chain
    - lowresfilter: 'int'. Residue lower threshold. Minimum number of residues per chain
    - verbose: If True get additional details (default False)
    """
    out_dir=checkPath(out_dir)
    PDBnames=[os.path.basename(pdb) for pdb in pdbs]
    if delimiter!=None:
        IDs=[ pdbname.replace(".pdb","").split(delimiter)[0] for pdbname in PDBnames]
    else:
        IDs=[ pdbname.replace(".pdb","") for pdbname in PDBnames]
    for i,pdb in enumerate(pdbs):
        pdbname=PDBnames[i]
        if verbose: print(pdbname)
        structurePDB=parsePDB('%s'%(pdb)) #From PDB to prody atom class
        hvPDB=structurePDB.getHierView() #Get prody hierarchical view
        chains=[chain.getChid() for chain in hvPDB] #Get chain identifiers
        for chain in chains:
            if verbose: print(chain)
            nresidues= hvPDB[chain].numResidues()
            if upresfilter!=None:
                if nresidues > upresfilter:
                    print("Too many residues for chain %s in %s"%(chain,pdbname))
                    continue
            if lowresfilter!=None:
                if nresidues < lowresfilter:
                    print("Too few residues for chain %s in %s"%(chain,pdbname))
                    continue
            writePDB('%s%s_%s.pdb'%(out_dir, IDs[i], chain), hvPDB[chain].select('protein')) #Save only protein

def splitPDBs(pdb_dir, schrodinger_path, out_dir=None, verbose=False):
    """
    SCHRODINGER LISENCE DEPENDENT
    Split PDBs from a directory into the target structure (receptor), the ligands,
    the waters and the ions/cofactors of the list of pdb files.
    - pdb_dir: directory with PDBs
    - out_dir: output directory
    - schrodinger_path: path to schrodinger
    -
    """
    exe_dir=os.getcwd()
    pdb_dir=checkPath(pdb_dir)
    if out_dir!=None: out_dir=checkPath(out_dir)
    os.chdir(pdb_dir)
    PDBs=glob.glob("*.pdb*")
    for PDB in PDBs:
        if '.pdb.gz' in PDB:
            cmd1="gunzip %s"%PDB
            os.system(cmd1)
            PDB=PDB.replace('.pdb.gz','.pdb')
            ID=PDB.replace(".pdb","")
        elif '.pdb' in PDB:
            ID=PDB.replace(".pdb","")
        if verbose: print('Extracting %s ligands...' %(ID))
        cmd2='%srun split_structure.py -m pdb %s %s.pdb -many_files'%(
            schrodinger_path, PDB, ID)
        os.system(cmd2)
    _organizeSplitOutput(pdb_dir=pdb_dir, exe_dir=exe_dir, out_dir=out_dir)

def _organizeSplitOutput(pdb_dir, exe_dir, out_dir=None):
    """
    SCHRODINGER LISENCE DEPENDENT
    Organize all the output files from splitPDBs (target/receptor, ligands, water and ions)
    into different directories
    - pdb_dir: directory with PDBs after the split
    - out_dir: output directory (if None then out_dir=pdb_dir)
    """
    os.chdir(exe_dir)
    if out_dir==None:
        out_dir=pdb_dir
    else:
        our_dir=checkPath(out_dir)
    os.mkdir("%sligand"%out_dir)
    os.mkdir("%sreceptor"%out_dir)
    os.mkdir("%swater"%out_dir)
    os.mkdir("%scof_ion"%out_dir)

    files=glob.glob("%s/*.pdb"%pdb_dir)
    in_dir=checkPath(os.path.dirname(files[0]))
    files=[os.path.basename(f) for f in files]

    for filename in files:
        if re.search("ligand", filename):
            shutil.move("%s%s"%(in_dir,filename), "%sligand/%s"%(out_dir,filename))
        if re.search("receptor", filename):
            shutil.move("%s%s"%(in_dir,filename), "%sreceptor/%s"%(out_dir,filename))
        if re.search("water", filename):
            shutil.move("%s%s"%(in_dir,filename), "%swater/%s"%(out_dir,filename))
        if re.search("cof_ion", filename):
            shutil.move("%s%s"%(in_dir,filename), "%scof_ion/%s"%(out_dir,filename))

def superimposePDB(mob_pdb, fix_pdb, out_dir, verbose=False):
    """
    Superimpose a mobile PDB into a fix PDB with Prody
    - mob_pdb: Mobile PDB to superimpose
    - fix_pdb: Fix PDB
    - out_dir: output directory
    - verbose: If True get additional details (default False)
    """
    structureFIX=parsePDB(fix_pdb)
    structureMOB=parsePDB(mob_pdb)
    nameMOB=os.path.basename(mob_pdb).replace(".pdb","")
    try:
        matchAlign(structureMOB,structureFIX)
    except:
        print('Superimposition between %s and %s failed'%(mob_pdb,fix_pdb))
        return -1
    writePDB('%s%s_super.pdb'%(out_dir,nameMOB),structureMOB)
    return 0

def superimposePDBs(pdbs, fix_pdb, out_dir, verbose=False):
    """
    Given a list of mobile PDBs and a fix superimpose all mobile elements to the fix PDB
    - pdbs: 'list'. PDBs list of mobile elements
    - fix_pdb: 'str'. Single PDB which will be fixed during the multiple superimpositions
               (fix element)
    - out_dir: 'str'. Directory to store all the superimposed PDBs
    - verbose: If True get additional details (default False)
    """
    out_dir=checkPath(out_dir)
    errors=0
    name_fix=os.path.basename(fix_pdb)
    for pdb in pdbs:
        name_mobile=os.path.basename(pdb)
        if verbose:
            print('Superimposition between fix PDB %s and mobile PDB %s'%(name_fix,name_mobile))
        aux_error=superimposePDB(pdb,fix_pdb,out_dir,verbose)
        errors += aux_error
    if verbose:  print('%d couldn\'t be superimpose'%-errors)

    #for i, pdb in enumerate(pdbs):
    #    #Superimpose using TMalign
    #    pdbname=PDBnames[i]
    #    if verbose: print(pdbname)
    #    outname='%s/%s'%(outdir,IDs[i])
    #    outname=outname.replace('//','/')
    #    TMalign_cmd='TMalign %s %s -o %s'%(pdb, fix_pdb,outname)
    #    print(TMalign_cmd)
    #    os.system(TMalign_cmd)
    #    os.system('rm %s %s_all %s_all_atm_lig %s_atm'%(outname,outname,outname,outname))
    #    os.system('mv %s_all_atm %s_all_atm.pdb'%(outname,outname))
    #
    #    #Extract mobile pdb from the TMalign output
    #    structurePDB=parsePDB('%s_all_atm.pdb'%(outname))
    #    #Load Prody Hierarchical Views
    #    hvPDB=structurePDB.getHierView()
    #    #Write superimpose mobile element as a pdb
    #    aux='%s/%s_super.pdb'%(outdir,IDs[i])
    #    aux=aux.replace("//","/")
    #    writePDB(aux,hvPDB['A'])
    #    os.system('rm %s_all_atm.pdb'%(outname))

def runPrepWizard(pdbs, schroodinger_path, delimiter=None, out_fmt='mae',
                  max_processes=4):
    """
    SCHRODINGER LISENCE DEPENDENT
    Given a list of pdbs, use PrepWizard from the Schroodinger premium package to prepare
    the PDB (protonate, add missing side chains etc...)
    - pdbs: 'list'. PDB list of elements to prepare
    - schroodinger_path: 'str'. Global path of Schroodinger
    - delimiter: 'str'. Delimiter to obtain an identifier from each PDB name
    - outfmt: 'str'. Outfile format. Either .mae or .pdb
    - max_processes: Number of processors used to paralalize the different executions
    """
    if out_fmt!='mae' and out_fmt!='pdb':
        raise ValueError('out_fmt must be either mae or pdb')
    PDBnames=[os.path.basename(pdb) for pdb in pdbs]
    if delimiter!=None:
        IDs=[ pdbname.replace(".pdb","").split(delimiter)[0] for pdbname in PDBnames]
    else:
        IDs=[ pdbname.replace(".pdb","") for pdbname in PDBnames]

    cmd_prepW=['%s/utilities/prepwizard -fillsidechains -WAIT %s %s_prep.%s'%(schroodinger_path,pdb,IDs[i],out_fmt) for i,pdb in enumerate(pdbs)]
    cmd_prepW=[cmd.replace('//','/') for cmd in cmd_prepW]
    processes=set()

    for cmd in cmd_prepW:
        print(cmd)
        processes.add(subprocess.Popen(cmd,shell=True))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

def cleanPrepWizard(out_dir, out_fmt='mae'):
    """
    SCHRODINGER LISENCE DEPENDENT
    Move the PrepWizard output to an specified directory
    - out_dir: 'str'. Output directory
    - out_fmt: 'str'. Outfile format of the PrepWizard. Either .mae or .pdb
    """
    out_dir=checkPath(out_dir)
    if out_fmt!='mae' and out_fmt!='pdb':
        raise ValueError('out_fmt must be either mae or pdb')
    os.system('mv *_prep.%s %s'%(out_fmt,out_dir))
    logdir='%slogs'%out_dir
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    if out_fmt=='mae': nout='pdb'
    else: nout='mae'
    try:
        os.system('mv *.%s %s'%(nout,logdir))
    except: pass
    try:
        os.system('mv *.log %s'%logdir)
    except: pass
    try:
        os.system('rm -r *-001')
    except: pass

def runSiteMap(maes, asl, schroodinger_path, delimiter=None, max_processes=4):
    """
    SCHRODINGER LISENCE DEPENDENT
    Run a SiteMap calculation for a list of MAEs (can't be pdbs).
    - maes: 'list'. MAE list of elements
    - asl: 'str'. ASL (atom specification Language)
    - schroodinger_path: 'str'. Global path of Schroodinger
    - delimiter: 'str'. Delimiter to obtain an identifier from each MAE name
    - max_processes: Number of processors used to paralalize the different executions
    """
    MAEnames=[os.path.basename(mae) for mae in maes]
    if delimiter!=None:
        IDs=[ maename.replace(".mae","").split(delimiter)[0] for
             maename in MAEnames]
    else:
        IDs=[ maename.replace(".mae","") for maename in MAEnames]

    cmd_SiteMap=['%s/sitemap -j %s -prot %s -sitebox 12 -resolution standard -reportsize 100 -writestructs no -maxsites 1 -siteasl "%s" -WAIT'%(schroodinger_path,IDs[i],mae,asl) for i,mae in enumerate(maes)]
    cmd_SiteMap=[cmd.replace("//","/") for cmd in cmd_SiteMap]

    processes=set()

    for cmd in cmd_SiteMap:
        print(cmd)
        processes.add(subprocess.Popen(cmd,shell=True))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

def _cleanSiteMap(out_dir, out_fmt='maegz'):
    """
    SCHRODINGER LISENCE DEPENDENT
    Move the SiteMap output to an specified directory
    - out_dir: 'str'. Output directory
    - out_fmt: 'str'. Outfile format of the PrepWizard. Either .mae or .pdb
    """
    out_dir=checkPath(out_dir)
    if out_fmt!='maegz':
        raise ValueError('out_fmt must be maegz')
    os.system('mv *.%s %s'%(out_fmt,out_dir))
    logdir='%slogs'%out_dir
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    os.system('mv *.vis %s'%logdir)
    os.system('mv *.smap %s'%logdir)
    os.system('mv *.log %s'%logdir)

def _groupSiteMap(sites, out_name, out_dir, schroodinger_path):
    """
    SCHRODINGER LISENCE DEPENDENT
    Group all volume sites from SiteMap into a single mae file
    - sites: list of siteMap files
    - out_name: output file name
    - out_dir: output file directory
    - schroodinger_path: 'str'. Global path of Schroodinger
    """
    conc_sites=''
    for site in sites:
        conc_sites=conc_sites + ' ' + site
    out_dir=checkPath(out_dir)
    cmd='%sutilities/structcat -imae %s -omae %s'%(schroodinger_path,conc_sites,out_name)
    os.system(cmd)
    try:
        aux='mv %s %s'%(out_name,out_dir)
        os.system(aux)
    except:
        print('wrong out_dir')

def getVolumeOverlapMatrix(sites, out, schroodinger_path, max_processes=4):
    """
    SCHRODINGER LISENCE DEPENDENT
    Generate pairwise volume overlap matrix
    - sites: 'str'. single file containing multiple SitMap files
    - out: output directory
    - schroodinger_path: 'str'. Global path of Schroodinger
    - max_processes: Number of processors used to paralalize the different executions
    """
    cmd='%srun volume_cluster.py -j %s -HOST localhost:%d -sc -r 2 %s'%(schroodinger_path,out,max_processes,sites)
    os.system(cmd)

def _cleanVolumeOverlapMatrix(matrix,out_dir):
    """
    SCHRODINGER LISENCE DEPENDENT
    Move the VolumeMatrix output to an specified directory
    out: 'str' Volume Overlapping Matrix file  (in csv)
    out_dir: 'str'. Output directory
    """
    out_dir=checkPath(out_dir)
    os.system('mv *.csv %s'%(out_dir))
    logdir='%slogs/'%out_dir
    if not os.path.isdir(logdir):
        os.system('mkdir %s'%logdir)
    os.system('mv *.mae %s'%logdir)
    os.system('mv *.log %s'%logdir)

def _uncompressMaegz(maegz, schroodinger_path):
    """
    SCHRODINGER LISENCE DEPENDENT
    inp:
    schroodinger_path: 'str'. Global path of Schroodinger
    """
    out=maegz.replace('.maegz','.mae')
    cmd='%sutilities/structcat -imae %s -omae %s'%(schroodinger_path,maegz,out)
    os.system(cmd)


class VolumeOverlappingMatrix(object):
    """""
    Class to store and analyse Volume Overlapping Matrix obtained with SiteMap from Schrodinger
    """""
    def __init__(self, csv, IDs=None, IDs_shroodinger=None, identifier=None,
                 delimiter='', del_position=0):
        """
        - csv: 'str'. csv file containing the volume overlapping matrix
        - IDs: 'list'. list of IDs to replace csv indices
        - IDs_shroodinger: 'str'. Input file used to compute the volume overlpaping matrix
                         with SiteMap
        """
        self.matrix=pd.read_csv(csv,delimiter=',',index_col=0)
        if IDs!=None and IDs_shroodinger==None:
            if len(IDs)!=self.matrix.shape[0]:
                raise ValueError('IDs should have the same length as rows/cols of csv file')
            self.IDs=IDs
            self.matrix=self.matrix.set_axis(self.IDs, axis=1)
            self.matrix=self.matrix.set_axis(self.IDs, axis=0)
        elif IDs==None and IDs_shroodinger!=None:
            if identifier==None:
                raise ValueError('To select the IDs from IDs_shroodinger an identifier is needed')
            self._get_IDs_shroodinger(IDs_shroodinger,identifier,delimiter,del_position)
        if IDs!=None and IDs_shroodinger!=None:
            raise ValueError('Pass either an IDs list or a IDs_shroodinger file')

    def _get_IDs_shroodinger(self, vm_input, identifier, delimiter='', del_position=0):
        """
        Get the IDs needed for the volume overlapping matrix from the input file used
        to compute it.
        - vm_input: 'str'. Input file used to compute the volume overlapping matrix (mae)
        - identifier: 'str'.
        """
        f=open(vm_input,'r')
        IDs=[]
        for line in f:
            if identifier in line:
                ID=line.split(delimiter)[del_position]
                if ID not in IDs:
                    IDs.append(ID)
        self.matrix=self.matrix.set_axis(axis=1, labels=IDs)
        self.matrix=self.matrix.set_axis(axis=0, labels=IDs)
        self.IDs=IDs

    def plot_hierarchical(self, out, fontsize=1):
        """

        - out:
        - fontsize: (default 1)
        """
        sns.set(font_scale=fontsize)
        cg=sns.clustermap(self.matrix, cmap="RdBu_r", yticklabels=True, xticklabels=True,
                          vmin=0,vmax=1)
        plt.savefig(out,dpi=300)

    def plot_hierarchical_labeled(self, properties_df, features, out, fontsize=1,
                                  printlabels=False, ucolors=None):
        """
        Hierarchical clustermap with color and row coloring according to a given feature of
        properties_df. 'pandas DataFrame'
        - features. 'list'. Columns/properties to be used during the coloring
        - out. 'str'. Outname
        - fontsize. 'int'. Fontsize
        - ucolors. 'list'.
        """
        Ncolors=0
        for feature in features:
            Ncolors += len(properties_df[feature].unique())
        if ucolors==None:
            rgb_colors=distinctipy.get_colors(Ncolors)
        else:
            if len(ucolors)!=Ncolors: raise ValueError('Incorrect number of colors')
            else:
                rgb_colors=ucolors
        columns=self.matrix.columns.tolist()
        colorindex=0
        list_dfcolors=[]
        list_luts=[]
        for i,feature in enumerate(features):
            colors=[]
            nflavours=len(properties_df[feature].unique())
            lut=dict(zip(properties_df[feature].unique(),rgb_colors[colorindex:colorindex+nflavours]))
            list_luts.append(lut)
            colorindex += nflavours
            print(lut)
            for j,column in enumerate(columns):
                ptype=properties_df.loc[column,feature]
                colors.append(lut[ptype])
            _dfcolors=pd.DataFrame({feature: colors}, index=columns)
            list_dfcolors.append(_dfcolors)
        dfcolors=pd.concat(list_dfcolors,axis=1)

        sns.set(font_scale=fontsize)
        cg=sns.clustermap(self.matrix,cmap="RdBu_r", row_colors=dfcolors, col_colors=dfcolors, yticklabels=printlabels,xticklabels=printlabels,vmin=0,vmax=1)

        #Plot legend
        for i,feature in enumerate(features):
            for flavour in properties_df[feature].unique():
                cg.ax_col_dendrogram.bar(0, 0, color=list_luts[i][flavour], label=flavour, linewidth=0)
                cg.ax_col_dendrogram.legend(loc='lower left', bbox_to_anchor=(0.9, 0.5) ,ncol=1)

        plt.savefig(out+'.pdf',dpi=300)


    def get_dendrogram(self,verbose=False):
        """

        -verbose:
        """
        dendrogram=linkage(self.matrix, 'average',metric='euclidean')
        self.dendrogram=dendrogram
        c, coph_dists=cophenet(self.dendrogram, pdist(self.matrix))
        self.c=c
        self.coph_dists=coph_dists
        if verbose:
            print('Cophenetic Correlation Coefficient: ', str(c))

    def _fancy_dendrogram(self,*args, **kwargs):
        """

        """
        max_d=kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold']=max_d
        annotate_above=kwargs.pop('annotate_above', 0)
        ddata=dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x=0.5 * sum(i[1:3])
                y=d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

    def plot_dendrogram(self, out, max_d, annotate_above, p=None):
        """
        - out:
        - max_d:
        - annotate_above:
        - p: (default None)
        """
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index',fontsize=16)
        plt.ylabel('distance',fontsize=16)
        if p==None:
            self._fancy_dendrogram(self.dendrogram, leaf_rotation=90., leaf_font_size=8.,
                                   labels=self.IDs, max_d=max_d, annotate_above=annotate_above)
        else:
            self._fancy_dendrogram(self.dendrogram, leaf_rotation=90., leaf_font_size=8.,
                                   labels=self.IDs, max_d=max_d, annotate_above=annotate_above,
                                   p=p,truncate_mode='lastp',show_contracted=True)
        ax=plt.gca()
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='major', labelsize=14)
        plt.savefig(out,dpi=300)

    def get_dendrogram_clusters(self,max_d):
        """

        - max_d:
        """
        clusters=fcluster(self.dendrogram, max_d, criterion='distance')
        self.clusters=clusters
        clusters_dic={}
        for i,id in enumerate(self.IDs):
            cluster=self.clusters[i]
            if cluster not in clusters_dic:
                clusters_dic[cluster]=[id]
            else:
                clusters_dic[cluster].append(id)
        self.clusters_dic=clusters_dic

    def save_dendrogram_clusters(self,out,verbose=False):
        """

        - out:
        - verbose:
        """
        if verbose:
            print(self.clusters_dic)
            for cluster in self.clusters_dic.keys():
                print(cluster,len(self.clusters_dic[cluster]))

        with open(out+'.p', 'wb') as handle:
            pickle.dump(self.clusters_dic, handle)

    def get_dendrogram_clustersCenters(self):
        """

        """
        clustersCenters={}
        for cluster in self.clusters_dic.keys():
            cluster_elements=self.clusters_dic[cluster]
            volume_matrix_cluster=self.matrix.loc[cluster_elements,cluster_elements]
            volume_matrix_cluster['mean']=volume_matrix_cluster.mean(axis=1)
            center=volume_matrix_cluster['mean'].idxmax()
            print('center cluster%d: '%cluster, center)
            clustersCenters[cluster]=center
        self.clustersCenters=clustersCenters

    def remove_cluster_outliers(self, list_clusters):
        """

        - list_clusters:
        """
        outlayer_clusters=list_clusters

        for cluster in self.clusters_dic.keys():
            if cluster in outlayer_clusters:
                #print(cluster)
                outlayers=self.clusters_dic[cluster]
                #print(outlayers)
                self.matrix.drop(labels=outlayers,axis=1,inplace=True)
                self.matrix.drop(labels=outlayers,axis=0,inplace=True)
                for outlayer in outlayers:
                    self.IDs.remove(outlayer)

    def organize_cluster_structures(self, results_dir, struct_dir, site_dir):
        """

        - results_dir:
        - struct_dir:
        - site_dir:
        """
        if not os.path.isdir('%s/clusters'%results_dir):
            os.system('mkdir %s/clusters'%results_dir)

        for cluster in self.clusters_dic.keys():
            cluster_elements=self.clusters_dic[cluster]
            for struct in cluster_elements:
                element_site='%s/Mpro_%s_super_out.maegz'%(site_dir, struct)
                element_prep="%s/Mpro_%s_super_prep.mae"%(struct_dir, struct)
                if not os.path.isdir('%s/clusters/%d'%(results_dir, cluster)):
                    os.system('mkdir %s/clusters/%d'%(results_dir, cluster))
                cmd1='cp %s %s/clusters/%d'%(element_site, results_dir, cluster)
                cmd2='cp %s %s/clusters/%d'%(element_prep, results_dir, cluster)
                os.system(cmd1)
                os.system(cmd2)

    def get_closest_elements_cluster_center(self, num_elements):
        """

        - num_elements:
        """
        for cluster in self.clusters_dic.keys():
            print('Cluster %s:' %cluster)
            center=self.clustersCenters[cluster]
            elements=self.clusters_dic[cluster]
            submatrix=self.matrix.loc[elements,elements]
            print(submatrix.nlargest(num_elements, [center])[center])
