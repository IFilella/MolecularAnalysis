# MolecularAnalysis

MolecularAnalysis is a Python library with valuable tools designed to facilitate **molecular analysis** tasks in the field of chemistry and bioinformatics. This library provides a wide range of tools and functions for handling molecular data.

It contains 4 different modules

## mollib.py
This module contains 2 main classes: **Mol** and **MolDB**

- **Mol**: This class can upload molecules in different formats (mol, smi, sdf, etc.). 
The most important functions include: get all parameters of the molecules, save it in different formats, and get molecular fragments and its correspoding connection with the reference molecule.
- **MolDB**: This class can upload databases of molecules in differents formats (mol list, sdf, smi list, etc.).
The most important functions of this class include the ones present in **Mol** class but applied in all the database of molecules, as well as other funcions for the databse analysis such as filter by uniqueness.


## pdb_analysis.py

This module provide valuable tools for treating proteins in **PDB format**.

The functions present in this module allow us to extract the protein and ligands from the PDB complex, to superimpose the structures to a fix target, to prepare the proteins using Protein Preparation Wizard, etc.
It also includes a class named **VolumeOverlappingMatrix**, obtained from the calculation of the volume binding site of each protein (using SiteMap), that allow us analysis tools such as hierarchical clustering.
> Note: Some functions need the Schrödinger licence.

## plotlib.py

This module containt functions for molecular representation such as trimap, tSNE, and UMAP.
This functions can be personalized by changing some parameters.



## sts_analysis.py

This module contains functions for performing statistical analysis.
These funcions include a binary classification study, and ROC curve analysis.

