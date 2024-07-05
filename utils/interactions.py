#!/usr/bin/env python

import prolif as plf
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import sys
import numpy as np
import dill
import progressbar
import random

class InteractionFingerprints(object):
    """

    """
    def __init__(self,
                 pdbfile=None,
                 sdffile=None,
                 intfpsfile=None,
                 interactions=['Hydrophobic',
                               'HBDonor',
                               'HBAcceptor',
                               'PiStacking',
                               'Anionic',
                               'Cationic',
                               'CationPi',
                               'PiCation',
                               'XBAcceptor',
                               'XBDonor',
                               'MetalAcceptor',
                               'MetalDonor'],
                 test=False):
        """
        Class init

        Input:
        ----------
        pdbfile: string
            PDB file with protein structure
        sdffile: string
            SDF file with the docked ligands
        intpfsfile: string
            pickle file to load a precalculated InteractionFingerprints object
        interactions: list
            list of interactions to asses during Fingerprint\
            calculation
            Default = interactions=["Hydrophobic", "HBDonor",\
                                    "HBAcceptor", "PiStacking",\
                                    "Anionic", "Cationic", "CationPi",\
                                    "PiCation", "XBAcceptor",\
                                    "XBDonor", "MetalDonor", "MetalAcceptor"]
            Warning: 'VdWContact', 'EdgeToFace', 'FaceToFace' can be added
        test: boolean
            If true it runs a test for the class
            Default False

        >>> ifps = InteractionFingerprints('../tests/8C3U_prep.pdb', '../tests/IL1beta_ref_cpd.sdf')
        >>> len(ifps.int_fps.to_bitvectors())
        15
        >>> fp0 = ifps.int_fps.to_bitvectors()[0]
        >>> fp0.GetNumOffBits()
        4
        >>> fp0.GetNumOnBits()
        14
        """
        self.test = test
        if intfpsfile:
            self.load_pickle(intfpsfile)
        else:
            if not pdbfile or not sdffile:
                raise ValueError('If a InteractionFingerprints object is not provided\
                                  both a pdbfile and sdffile must be provided')
            # Load protein pdb and molecules sdf to prolif
            protein_rdkit = Chem.MolFromPDBFile(pdbfile, removeHs=False)
            protein_plf = plf.Molecule(protein_rdkit)
            sdf_plf = plf.sdf_supplier(sdffile)

            # Retrive molecules names from sdf
            mol_names = []
            mols = []
            for mol in sdf_plf:
                mols.append(mol)
                mol_names.append(mol.GetProp('_Name'))

            self.mol_names = mol_names
            self.mols = mols
            #self.mol_smiles = 

            # Compute inetraction fingerprints
            int_fps = plf.Fingerprint(interactions=interactions)
            int_fps.run_from_iterable(sdf_plf, protein_plf)
            self.int_fps = int_fps
            self.df_int = self.int_fps.to_dataframe()

    def load_pickle(self, intfpsfile):
        """

        """
        with open(intfpsfile, 'rb') as f:
            intfpsobject = dill.load(f)
            self.mol_names = intfpsobject.mol_names
            self.int_fps = intfpsobject.int_fps
            self.df_int = intfpsobject.df_int
            self.mols = intfpsobject.mols
        #self.int_fps = plf.Fingerprint.from_pickle(intfpsfile)
        #print(self.int_fps)

    def to_pickle(self, outname):
        """

        """
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with open(outname+'.intfp', 'wb') as handle:
            dill.dump(self, handle)
        #self.int_fps.to_pickle('%s.intfp' % outname)

    def to_csv(self, outname):
        """

        """
        df = self.int_fp.to_dataframe()
        df.to_csv(outname)

    def get_numdiff_interactions(self):
        """

        """
        int_fps_vecs = self.int_fps.to_countvectors()
        int_fps_bitvecs = [''.join(map(str,int_fp.ToList())) for int_fp in int_fps_vecs]
        numdiff_interactions = len(set(int_fps_bitvecs))
        print(numdiff_interactions)
        return numdiff_interactions

    def get_int_perc(self, plot=None):
        """

        """
        df_int = self.df_int.droplevel('ligand', axis=1)
        df_int_perc = df_int.mean(axis=0)
        self.df_int_perc = df_int_perc
        print(self.df_int_perc)
        if plot:
            df_unstack = self.df_int_perc.unstack(level='interaction')
            df_unstack = df_unstack.fillna(0)
            print(df_unstack)
            plt.figure(dpi=300, figsize=(10, 7))
            sns.heatmap(df_unstack, annot=df_unstack.round(4), cmap='viridis')
            plt.savefig(plot)

    def get_int_res_perc(self, plot=None):
        """

        """
        df_int = self.df_int.droplevel('ligand', axis=1)
        df_int_res_perc = df_int.droplevel('interaction', axis=1)
        df_int_res_perc = df_int_res_perc.groupby(level=0, axis=1).any()
        mean_values = df_int_res_perc.mean(axis=0)
        df_int_res_perc = pd.DataFrame(mean_values, columns=['Percentage'])
        self.df_int_res_perc = df_int_res_perc
        print(self.df_int_res_perc)
        if plot:
            plt.figure(dpi=300)
            sns.heatmap(self.df_int_res_perc,
                        annot=self.df_int_res_perc.round(4),
                        cmap='viridis')
            plt.savefig(plot)

    def plot_pair_similarity_dist(self, outname, n=10000000):
        """
        Structural similarity Vs Interaction similarity
        """
        # Get Morgan Fingerprints
        str_fps_vecs = []
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)
        print('Computing structural fingerprints')
        bar = progressbar.ProgressBar(maxval=len(self.mols)).start()
        for i, mol in enumerate(self.mols):
            bar.update(i)
            str_fps_vecs.append(fpgen.GetSparseCountFingerprint(mol))
        bar.finish()

        # Get structural similarit matrix
        str_similarity_matrix = []
        print('Computing structural similarity matrix')
        bar = progressbar.ProgressBar(maxval=len(str_fps_vecs)).start()
        for i, str_fp in enumerate(str_fps_vecs):
            bar.update(i)
            str_similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(str_fp,
                                                                            str_fps_vecs))
        bar.finish()

        # Get interaction similarity matrix
        int_fps_vecs = self.int_fps.to_countvectors()
        int_similarity_matrix = []
        print('Computing interaction similarity matrix')
        bar = progressbar.ProgressBar(maxval=len(int_fps_vecs)).start()
        for i, cv in enumerate(int_fps_vecs):
            bar.update(i)
            int_similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(cv,
                                                                            int_fps_vecs))
        bar.finish()

        # Plot
        x_vals = [item for sublist in str_similarity_matrix for item in sublist]
        y_vals = [item for sublist in int_similarity_matrix for item in sublist]

        indices = list(range(len(x_vals)))
        random.shuffle(indices)
        indices = indices[:n]

        _x_vals = [x_vals[i] for i in indices]
        _y_vals = [y_vals[i] for i in indices]

        plt.figure(dpi=300)
        plt.scatter(_x_vals, _y_vals, marker='.', s=1)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Structural Tanimoto Similarity')
        plt.ylabel('Interactions Tanimoto Similarity')
        plt.savefig(outname)

    def plot_barcode(self, outname, mol_labels=None):
        """
        Extracted from prolig source code (Jul 2024)
        """
        # Import color dictionary of prolif
        from prolif.plotting.utils import separated_interaction_colors
        separated_interaction_colors[None] = "white"

        if not mol_labels:
            mol_labels = self.mol_names
        color_mapper = {interaction: value for value, interaction in enumerate(separated_interaction_colors)}
        inv_color_mapper = {value: interaction for interaction, value in color_mapper.items()}
        cmap = ListedColormap(list(separated_interaction_colors.values()))

        n_ligand_residues = len(np.unique(self.df_int.columns.get_level_values("ligand")))
        if n_ligand_residues == 1:
            df = self.df_int.droplevel('ligand', axis=1)

        def _bit_to_color_value(s: pd.Series) -> pd.Series:
            """Replaces a bit value with it's corresponding color value"""
            interaction = s.name[-1]
            return s.apply(
                lambda v: (
                    color_mapper[interaction] if v else color_mapper[None]
                )
            )

        df = df.astype(np.uint8).T.apply(_bit_to_color_value, axis=1)
        fig, ax = plt.subplots(dpi=300, figsize=(8, 10))
        ax: plt.Axes
        im = ax.imshow(
            df.values,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=max(color_mapper.values()),
        )

        # Frame ticks
        frames = df.columns
        max_ticks = len(frames) - 1
        n_frame_ticks = len(df.columns)
        # try to have evenly spaced ticks
        for effective_n_ticks in (n_frame_ticks, n_frame_ticks - 1, n_frame_ticks + 1):
            samples, step = np.linspace(0, max_ticks, effective_n_ticks, retstep=True)
            if step.is_integer():
                break
        else:
            samples = np.linspace(0, max_ticks, n_frame_ticks)
        indices = np.round(samples).astype(int)
        ax.xaxis.set_ticks(indices, frames[indices])
        ax.xaxis.set_ticklabels(mol_labels, rotation='vertical', fontsize=6)
        ax.set_xlabel('Molecules')

        # Residues ticks
        n_items = len(df.index)
        residues = df.index.get_level_values("protein")
        interactions = df.index.get_level_values("interaction")
        indices = [
            i
            for i in range(n_items)
            if (i - 1 >= 0 and residues[i - 1] != residues[i]) or i == 0
        ]
        ax.yaxis.set_ticks(indices, residues[indices])

        # legend
        values = np.unique(df.values).tolist()
        try:
            values.pop(-1)  # remove None color (the last one that we introduced into the dict)
        except ValueError:
            # 0 not in values (e.g. plotting a single frame)
            pass
        legend_colors = {
            inv_color_mapper[value]: im.cmap(value) for value in values
        }
        patches = [
            Patch(color=color, label=interaction)
            for interaction, color in legend_colors.items()
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
        fig.tight_layout(pad=1.2)
        plt.savefig(outname)

    def plot_clustermap(self, outname, mol_labels=None):
        """

        """
        countvectors = self.int_fps.to_countvectors()
        similarity_matrix = []
        bar = progressbar.ProgressBar(maxval=len(countvectors)).start()
        for i, cv in enumerate(countvectors):
            bar.update(i)
            print('%d/%d' % (i, len(countvectors)))
            similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(cv,
                                                                        countvectors))
        bar.finish()
        if mol_labels:
            similarity_matrix = pd.DataFrame(similarity_matrix,
                                             index=mol_labels,
                                             columns=mol_labels)
        else:
            similarity_matrix = pd.DataFrame(similarity_matrix,
                                             index=self.mol_names,
                                             columns=self.mol_names)
        plt.subplots(dpi=300, layout='constrained')
        g = sns.clustermap(similarity_matrix, cmap='PRGn')
        ax = g.ax_heatmap
        ax.set_xlabel('Molecule')
        ax.set_ylabel('Molecule')
        plt.savefig(outname)


if __name__ == "__main__":
    import doctest
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test',
                        help='Test the code',
                        action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(
            optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    protein_file = '../tests/8C3U_prep.pdb'
    sdf_file = '../tests/IL1beta_ref_cpd.sdf'

    ifps = InteractionFingerprints(protein_file, sdf_file)
    ifps.plot_clustermap(outname='clustermap.pdf')
    ifps.plot_barcode(outname='interactions.png')
