import rdkit.Chem.BRICS as BRICS
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import re
import warnings

def read_compoundDB(data):
    compoundDB = Chem.SDMolSupplier(data)
    return compoundDB


def get_MolSimilarity(mol1,mol2,metric='Tanimoto'):
    fp1 = Chem.RDKFingerprint(mol1.mol)
    fp2 = Chem.RDKFingerprint(mol2.mol)
    if metric == 'Tanimoto':
        return DataStructs.TanimotoSimilarity(fp1,fp2)
    elif  metric == 'Dice':
        return DataStructs.DiceSimilarity(fp1,fp2)
    elif metric == 'Cosine':
        return DataStructs.CosineSimilarity(fp1,fp2)
    elif metric == 'Sokal':
        return DataStructs.SokalSimilarity(fp1,fp2)
    elif metric == 'Russel':
        return DataStructs.RusselSimilarity(fp1,fp2)
    elif metric == 'Kulczynski':
        return DataStructs.KulczynskiSimilarity(fp1,fp2)
    elif metric == 'McConnaughey':
        return DataStructs.McConnaugheySimilarity(fp1,fp2)
    else:
        raise ValueError('Invalid Metric')

def get_no_anchoring_frag(frag):
    noanfrag = re.sub("\[.*?\]", "", frag)
    noanfrag = re.sub("\(\)","",noanfrag)
    return noanfrag

class mol(object):
    """"""
    """"""
    def __init__(self ,smile = None, InChI = None):
        if smile != None and InChI == None:
            self.smile = smile
            self.mol = Chem.MolFromSmiles(self.smile)
        elif smile == None and InChI != None:
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
        elif smile != None and InChI != None:
            warnings.warn(f'Given that both SMILE and InChI format have been provided the molecule will be read with InChI')
            self.smile = smile
            self.InChI = InChI
            self.mol = Chem.MolFromInchi(self.InChI)
        else:
            raise ValueError('To initialize the molecule SMILE or InChI format is required')


    def get_BRICSdecomposition(self):
        self.fragments = list(BRICS.BRICSDecompose(self.mol))

    def get_clean_fragments(self):
        if self.fragments == None: self.get_BRICSdecomposition()
        self.cfragments = [re.sub("(\[.*?\])", "[*]", frag) for frag in self.fragments]
        
if __name__ == '__main__':
    smile = 'Cc1cc(-c2csc(N=C(N)N)n2)cn1C'
    inchi = 'InChI=1S/C10H13N5S/c1-6-3-7(4-15(6)2)8-5-16-10(13-8)14-9(11)12/h3-5H,1-2H3,(H4,11,12,13,14)'
    #Testing init
    m = mol(smile=smile)
    m = mol(InChI=inchi)
    m = mol(smile=smile,InChI=inchi)
    #Testing BRICS decomposition
    m.get_BRICSdecomposition()
    m.get_clean_fragments()
    #Testing Tanimoto Similarity
    similarity = get_MolSimilarity(m,m)
    print(similarity)
