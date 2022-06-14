import rdkit.Chem.BRICS as BRICS
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import re
import warnings
import rdkit.Chem.Lipinski as Lipinski

def filter_db_similarity(indb,outdb,verbose=True):
    inp = open(indb,'r')
    out = open(outdb,'w')
    uniquefrags = {}
    count = 0
    for i, line in enumerate(inp):
        count+=1
        line = line.split()
        SMILE = line[0]
        IDs = line[-1]
        m1 = mol(smile=SMILE)
        m1_atoms = m1.mol.GetNumAtoms()
        m1_NOcount = Lipinski.NOCount(m1.mol)
        m1_NHOHcount = Lipinski.NHOHCount(m1.mol)
        m1_rings = Lipinski.RingCount(m1.mol)
        m1_sp3 = Lipinski.FractionCSP3(m1.mol)
        m1_NumAliphaticRings = Lipinski.NumAliphaticRings(m1.mol)
        m1_NumAromaticRings = Lipinski.NumAromaticRings(m1.mol)
        if len(uniquefrags.keys()) == 0:
            uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
        for j,k in enumerate(uniquefrags.keys()):
            m2 = uniquefrags[k][2]
            m2_atoms = uniquefrags[k][3]
            m2_NOcount = uniquefrags[k][4]
            m2_NHOHcount = uniquefrags[k][5]
            m2_rings = uniquefrags[k][6]
            m2_sp3 = uniquefrags[k][7]
            m2_NumAliphaticRings = uniquefrags[k][8]
            m2_NumAromaticRings = uniquefrags[k][9]
            if m1_atoms != m2_atoms or m1_NOcount != m2_NOcount or m1_rings != m2_rings or m1_NHOHcount != m2_NHOHcount or m1_sp3 != m2_sp3 or m1_NumAliphaticRings != m2_NumAliphaticRings or  m1_NumAromaticRings != m2_NumAromaticRings:
                if j == len(uniquefrags.keys())-1:
                    uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                    break
                else:
                    continue
            else:
                similarity =  get_MolSimilarity(m1,m2)
                if j == len(uniquefrags.keys())-1 and similarity != 1:
                    uniquefrags[SMILE] = [SMILE,IDs,m1,m1_atoms,m1_NOcount,m1_NHOHcount,m1_rings,m1_sp3,m1_NumAliphaticRings,m1_NumAromaticRings]
                    break
                elif j == len(uniquefrags.keys())-1 and similarity == 1:
                    print("0: " + uniquefrags[k][0] + " " +SMILE)
                    uniquefrags[k][0] += ',' + SMILE
                    uniquefrags[k][1] += ',' + IDs
                    break
                elif similarity == 1:
                    print("1: " + uniquefrags[k][0] + " " +SMILE)
                    uniquefrags[k][0] += ',' + SMILE
                    uniquefrags[k][1] += ',' + IDs
                    break
                else:
                    continue
        if verbose:
            print('Unique fragments: %d'%len(uniquefrags.keys()))
            print('Analized fragments: %d'%count)
        else:
            if count % 1000 == 0:  print('Analized fragments: %d'%count)

    inp.close()
    if verbose: print("Saving into file")
    for k in uniquefrags.keys():
        out.write(k + " " + uniquefrags[k][0] + " " +uniquefrags[k][1] + "\n")
    out.close()

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

class fragDB(object):
    """""
    """""
    def __init__(self, txtDB = None, dicDB = None):
        self.txtDB = txtDB
        self.dicDB = dicDB
        if self.txtDB != None and self.dicDB == None:
            self.dicDB = {}
            db = open(self.txtDB,'r')
            for line in db:
                line = line.split()
                SMILE = line[0]
                simSMILES = line[1]
                IDs = line[2]
                m1 = mol(smile=SMILE)
                m1_atoms = m1.mol.GetNumAtoms()
                m1_NOcount = Lipinski.NOCount(m1.mol)
                m1_NHOHcount = Lipinski.NHOHCount(m1.mol)
                m1_rings = Lipinski.RingCount(m1.mol)
                m1_sp3 = Lipinski.FractionCSP3(m1.mol)
                m1_NumAliphaticRings = Lipinski.NumAliphaticRings(m1.mol)
                m1_NumAromaticRings = Lipinski.NumAromaticRings(m1.mol)
                self.dicDB[SMILE] = [simSMILES,]
            

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
    
    def get_NumAtoms(self):
        self.NumAtoms = self.mol.GetNumAtoms()

    def get_NOCount(self):
        self.NOCount = Lipinski.NOCount(self.mol)

    def get_NHOHcount(self):
        self.NHOHcount = Lipinski.NHOHCount(self.mol)

    def get_rings(self):
        self.rings = Lipinski.RingCount(self.mol)

    def get_sp3(self):
        self.sp3 = Lipinski.FractionCSP3(self.mol)

    def get_NumAliphaticRings(self):
        self.NumAliphaticRings = Lipinski.NumAliphaticRings(self.mol)

    def get_NumAromaticRings(self):
        self.NumAromaticRings = Lipinski.NumAromaticRings(self.mol)
        
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
