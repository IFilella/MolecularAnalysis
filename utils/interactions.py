#!/usr/bin/env python

import prolif as plf
from rdkit import Chem
from rdkit import DataStructs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import sys
import numpy as np

class InteractionFingerprints(object):
    """

    """
    def __init__(self,
                 pdbfile,
                 sdffile,
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
        # Load protein pdb and molecules sdf to prolif
        protein_rdkit = Chem.MolFromPDBFile(pdbfile, removeHs=False)
        protein_plf = plf.Molecule(protein_rdkit)
        sdf_plf = plf.sdf_supplier(sdffile)

        # Retrive molecules names from sdf
        mol_names = []
        for mol in sdf_plf:
            mol_names.append(mol.GetProp('_Name'))

        self.mol_names = mol_names

        # Compute inetraction fingerprints
        int_fps = plf.Fingerprint(interactions=interactions)
        int_fps.run_from_iterable(sdf_plf, protein_plf)
        self.int_fps = int_fps
        self.int_df = self.int_fps.to_dataframe()

    def to_pickle(self, outname):
        """

        """
        self.int_fp.to_pickle(outname)

    def to_csv(self, outname):
        """

        """
        df = self.int_fp.to_dataframe()
        df.to_csv(outname)

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

        n_ligand_residues = len(np.unique(self.int_df.columns.get_level_values("ligand")))
        if n_ligand_residues == 1:
            df = self.int_df.droplevel('ligand', axis=1)

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
        for cv in countvectors:
            similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(cv,
                                                                        countvectors))
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
