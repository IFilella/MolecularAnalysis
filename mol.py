import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.Descriptors as Descriptors


class Mol(object):
    """
    Molecule class:
    - smile: to load the molecule oject from a SMILE str
    - InChi: to load the molecule oject from an InChi str
    - rdkit: to load the molecule oject from a rdkit Mol object
    - pdb: to load the molecule oject from a pdb file str
    - allparamaters: if True calculate multiple molecular
                     paramaters (default False)
    - chirality: if False remove chirality (default True)
    - name: name identifier for the molecule object
    """
    def __init__(self, smile=None, InChI=None, rdkit=None, pdb=None, allparamaters=False,
                 chirality=True, name=None):
        #Load from SMILE
        if smile!=None and InChI==None and rdkit==None and pdb==None:
            self.smile=smile
            self.InChi=None
            if not chirality:
                self.smile=self.smile.replace("@","")
            self.molrdkit=Chem.MolFromSmiles(self.smile)
            if self.molrdkit==None:
                self.error=-1
            else:
                self.error=0
        #Load from InChi
        elif smile==None and InChI!=None and rdkit==None and pdb==None:
            self.InChI=InChI
            self.molrdkit=Chem.MolFromInchi(self.InChI)
            if self.molrdkit==None:
                self.error=-1
            else:
                self.error=0
            self.smile=Chem.MolToSmiles(self.molrdkit)
            if not chirality:
                self.smile=self.smile.replace("@","")
        #Load from rdkit Mol object
        elif smile==None and InChI==None and rdkit!=None and pdb==None:
             self.molrdkit=rdkit
             if self.molrdkit==None:
                 self.error=-1
             else:
                 self.error=0
             self.smile=Chem.MolToSmiles(self.molrdkit)
             self.InChi=None
             if not chirality:
                 self.smile=self.smile.replace("@","")
        #Load from pdb file
        elif smile==None and InChI==None and rdkit==None and pdb!=None:
            self.molrdkit=Chem.MolFromPDBFile(pdb)
            if self.molrdkit==None:
                self.error=-1
            else:
                self.error=0
            self.smile=Chem.MolToSmiles(self.molrdkit)
            self.InChi=None
            if not chirality:
                self.smile=self.smile.replace("@","")
        else:
            raise AttributeError('Provide only a smile, a InchI, a rdkit mol or a pdb')
        #Calculate rdkit molecular paramaters such as NOCount, RingCount, ...
        if allparamaters:
            self.getAllParamaters()
        if name:
            self.name=name
            self.molrdkit.SetProp('_Name',self.name)
        else:
            self.name='unk'

    def saveToMol(self,output):
        """
        Save the mol object to mol format (without format)
        - output: output file name
        """
        file=open(output+'.mol','w+')
        file.write(Chem.MolToMolBlock(self.molrdkit))
        file.close()

    def saveToPDB(self,output):
        """
        Save the mol object to PDB format
        - output: output file name (without format)
        """
        file=open(output+'.pdb','w+')
        file.write(Chem.MolToPDBBlock(self.molrdkit))
        file.close()

    def getFingerPrint(self,alg='RDKIT',nBits=2048):
        """
        Get molecular FingerPrint
        - alg: Algrithm used to compute the Fingerprint (default Morgan4)
        - nBits: Number of bits of the Fingerprint (default 2048)
        """
        if alg == 'RDKIT':
            self.FingerPrint=Chem.RDKFingerprint(self.molrdkit)
        elif alg == 'Morgan2':
            self.FingerPrint=AllChem.GetMorganFingerprintAsBitVect(self.molrdkit, 1, nBits=2048)
        elif alg == 'Morgan4':
            self.FingerPrint=AllChem.GetMorganFingerprintAsBitVect(self.molrdkit, 2, nBits=2048)
        elif alg == 'Morgan6':
            self.FingerPrint=AllChem.GetMorganFingerprintAsBitVect(self.molrdkit, 3, nBits=2048)
        elif alg == 'Morgan8':
            self.FingerPrint=AllChem.GetMorganFingerprintAsBitVect(self.molrdkit, 4, nBits=2048)
        else:
            raise KeyError('Invalid fingerprint algorithm')
        return self.FingerPrint

    def getParamaters(self):
        """
        Get several molecular paramaters using rdkit functions
        """
        try:
            Chem.SanitizeMol(self.molrdkit)
        except:
            self.error=-1
            return
        self.get_NumAtoms()
        self.get_NOCount()
        self.get_NHOHCount()
        self.get_RingCount()
        self.get_sp3()
        self.get_NumAliphaticRings()
        self.get_NumAromaticRings()
        self.get_MolWt()
        self.get_LogP()
        self.get_NumHAcceptors()
        self.get_NumHDonors()
        self.get_NumHeteroatoms()
        self.get_NumRotatableBonds()
        self.get_NumHeavyAtoms()
        self.get_NumAliphaticCarbocycles()
        self.get_NumAliphaticHeterocycles()
        self.get_NumAromaticCarbocycles()
        self.get_NumAromaticHeterocycles()
        self.get_TPSA()
        if self.molrdkit.GetNumConformers()>0:
            self.get_NPR1()
            self.get_NPR2()
            self.get_InertialShapeFactor()
            self.get_RadiusOfGyration()

    def get_NumAtoms(self):
        try: self.NumAtoms=self.molrdkit.GetNumAtoms()
        except: self.NumAtoms=None

    def get_NOCount(self):
        try: self.NOCount=Lipinski.NOCount(self.molrdkit)
        except: self.NOCount=None

    def get_NHOHCount(self):
        try: self.NHOHCount=Lipinski.NHOHCount(self.molrdkit)
        except: self.NHOHCount=None

    def get_RingCount(self):
        try: self.RingCount=Lipinski.RingCount(self.molrdkit)
        except: None

    def get_sp3(self):
        try: self.FractionCSP3=Lipinski.FractionCSP3(self.molrdkit)
        except: None

    def get_NumAliphaticRings(self):
        try: self.NumAliphaticRings=Lipinski.NumAliphaticRings(self.molrdkit)
        except: None

    def get_NumAromaticRings(self):
        try: self.NumAromaticRings=Lipinski.NumAromaticRings(self.molrdkit)
        except: None

    def get_MolWt(self):
        try: self.MolWt=Descriptors.ExactMolWt(self.molrdkit)
        except: None

    def get_LogP(self):
        try: self.LogP=Chem.Descriptors.MolLogP(self.molrdkit)
        except: None

    def get_NumHAcceptors(self):
        try: self.NumHAcceptors=Chem.Descriptors.NumHAcceptors(self.molrdkit)
        except: None

    def get_NumHDonors(self):
        try: self.NumHDonors=Chem.Descriptors.NumHDonors(self.molrdkit)
        except: None

    def get_NumHeteroatoms(self):
        try: self.NumHeteroatoms=Chem.Descriptors.NumHeteroatoms(self.molrdkit)
        except: None

    def get_NumRotatableBonds(self):
        try: self.NumRotatableBonds=Chem.Descriptors.NumRotatableBonds(self.molrdkit)
        except: None

    def get_NumHeavyAtoms(self):
        try: self.NumHeavyAtoms=Chem.Descriptors.HeavyAtomCount(self.molrdkit)
        except: None

    def get_NumAliphaticCarbocycles(self):
        try: self.NumAliphaticCarbocycles=Chem.Descriptors.NumAliphaticCarbocycles(self.molrdkit)
        except: None

    def get_NumAliphaticHeterocycles(self):
        try: self.NumAliphaticHeterocycles=Chem.Descriptors.NumAliphaticHeterocycles(self.molrdkit)
        except: None

    def get_NumAromaticCarbocycles(self):
        try: self.NumAromaticCarbocycles=Chem.Descriptors.NumAromaticCarbocycles(self.molrdkit)
        except: None

    def get_NumAromaticHeterocycles(self):
        try: self.NumAromaticHeterocycles=Chem.Descriptors.NumAromaticHeterocycles(self.molrdkit)
        except: None

    def get_TPSA(self):
        try: self.TPSA=Chem.Descriptors.TPSA(self.molrdkit)
        except: None

    def get_NPR1(self):
        try: self.NPR1=Chem.rdMolDescriptors.CalcNPR1(self.molrdkit)
        except: None

    def get_NPR2(self):
        try: self.NPR2=Chem.rdMolDescriptors.CalcNPR2(self.molrdkit)
        except: None

    def get_InertialShapeFactor(self):
        try: self.InertialShapeFactor=Chem.Descriptors3D.InertialShapeFactor(self.molrdkit)
        except: None

    def get_RadiusOfGyration(self):
        try: self.RadiusOfGyration=Chem.Descriptors3D.RadiusOfGyration(self.molrdkit)
        except: None
