# MolecularAnalysis

MolecularAnalysis is a Python library with valuable tools designed to facilitate **molecular analysis** tasks in the field of chemistry and bioinformatics. This library provides a wide range of tools and functions for handling molecular data.

It consists on 2 main classes:
- **mol.Mol**: This class can upload molecules in different formats (rdkit mol, smile, pdb, etc.).
- **moldb.MolDB**: This class can upload databases of molecules in differents formats (rdkit mol list, sdf, smi, etc.).

It also contains multiple modules:
- **analysis.plot**: To plot molecular databases using dimensional reduction approaches (UMAP, TSNE, ...)
- **analysis.sts**: To performe multiple statistical analysis such as ROC plots or confusion plots
- **fragments.fragmentation**: to fragmentize molecules
- **utils.pdb**: This module allows the manipulation of proteins such as extracting the protein and ligands from the PDB complex, to superimpose the structures to a fix target, to prepare the proteins using Protein Preparation Wizard, etc.
> Note: Some functions in utils.pdb need a Schrödinger licence.

