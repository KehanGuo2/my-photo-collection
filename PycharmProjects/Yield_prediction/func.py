from rdkit import Chem
from openbabel import openbabel as ob
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem import rdPartialCharges as GMCharge
import numpy as np


def one_hot_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


class Atom_feat:

    def atom_type_one_hot(atom):
        """One hot encoding of atom type."""
      
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                        'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                        'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                        'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']

        return one_hot_encoding(atom.GetSymbol(), allowable_set)

    def atom_radical_electrons(atom):
        """Get number of radical electrons of an atom."""
        return [atom.GetNumRadicalElectrons()]

    def atom_explicit_valence(atom):
        """Get the explicit valence of an atom."""
        return [atom.GetExplicitValence()]

    def atom_hybridization_one_hot(atom):
        """Get the hybridization type of an atom."""
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2]
        return one_hot_encoding(atom.GetHybridization(), allowable_set)

    def atom_partial_charge(atom):
        """Get Gasteiger partial charge for an atom."""
        
        gasteiger_charge = atom.GetProp('_GasteigerCharge')
        if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
            gasteiger_charge = 0
        return [float(gasteiger_charge)]

    def atom_chirality_type_one_hot(atom):
        """One hot encoding for the chirality type of an atom."""
        if not atom.HasProp('_CIPCode'):
            return [False, False]
        allowable_set = ['R', 'S']
        return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set)
    feat_list = [atom_type_one_hot, atom_radical_electrons,atom_explicit_valence,atom_hybridization_one_hot,atom_partial_charge,atom_chirality_type_one_hot]

class Bond_feat:

    def bond_type_one_hot(bond):
        """One hot encoding for the type of a bond."""
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC]
        return one_hot_encoding(bond.GetBondType(), allowable_set)


    def bond_stereo_one_hot(bond):
        """One hot encoding for the stereo configuration of a bond."""
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                        Chem.rdchem.BondStereo.STEREOANY,
                        Chem.rdchem.BondStereo.STEREOZ,
                        Chem.rdchem.BondStereo.STEREOE,
                        Chem.rdchem.BondStereo.STEREOCIS,
                        Chem.rdchem.BondStereo.STEREOTRANS]
        return one_hot_encoding(bond.GetStereo(), allowable_set)

    
    feat_list = [bond_type_one_hot, bond_stereo_one_hot]


class Mol:

    def __init__(self, smiles=None, xyz=None, smarts=None):
        self.smiles = smiles
        self.xyz = xyz
        self.smarts = smarts
        if smiles !=None:
            mol = Chem.MolFromSmiles(smiles)
        elif smarts!=None:
            mol = Chem.MolFromSmarts(smarts)
            mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        GMCharge.ComputeGasteigerCharges(mol,12)
        self.mol = mol
    
    def get_atom_feature(self):
        mol = self.mol
        num_atoms = mol.GetNumAtoms()
        Mol_atoms = np.empty(num_atoms, dtype = object)
        print(num_atoms)
        mol.UpdatePropertyCache()
        self.mol = mol
        GMCharge.ComputeGasteigerCharges(mol,12)
        for i in range(num_atoms):
            atom_features = []
            atom = mol.GetAtomWithIdx(i)
            for feat_name in Atom_feat.feat_list:
                atom_features = np.append(atom_features,(feat_name(atom)))
            Mol_atoms[i] = atom_features
        return Mol_atoms


    def get_bond_feature(self):       
        mol = self.mol
        num_bonds = mol.GetNumBonds()
        adjacency = np.empty(num_bonds, dtype = object)
        Mol_bonds = np.empty(num_bonds, dtype = object)
        for i in range(num_bonds):
            bond_features = []
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            adjacency[i] = [u,v]
            for feat_name in Bond_feat.feat_list:
                bond_features = np.append(bond_features,(feat_name(bond)))
            Mol_bonds[i] = bond_features
        
        return adjacency,Mol_bonds

    
    def get_func(self):
        with open('patterns.txt','r') as f:
            lines = f.readlines()
        func_g = {x.split()[0]:x.split()[1] for x in lines}
        mol =self.mol
        func_list = []
        for fg in func_g.keys():
            func = {}
            fg_feat = [1,2,3,4]
            substruct = Chem.MolFromSmarts(fg)
            hit_ats = list(mol.GetSubstructMatch(substruct))
            if len(hit_ats)>0:
                func['label'] = func_g[fg]
                func['position'] = hit_ats
                func['feature'] = fg_feat
                func_list.append(func)
        return func_list

if __name__ == "__main__":
    reaction ='[CH2:23]1[O:24][CH2:25][CH2:26][CH2:27]1.[F:1][c:2]1[c:3]([N+:10](=[O:11])[O-:12])[cH:4][c:5]([F:9])[c:6]([F:8])[cH:7]1.[H-:22].[NH2:13][c:14]1[s:15][cH:16][cH:17][c:18]1[C:19]#[N:20].[Na+:21]>>[c:2]1([NH:13][c:14]2[s:15][cH:16][cH:17][c:18]2[C:19]#[N:20])[c:3]([N+:10](=[O:11])[O-:12])[cH:4][c:5]([F:9])[c:6]([F:8])[cH:7]1' 
    reactants = reaction.split('>>')[0]
    product = reaction.split('>>')[1]

    for reactant in reactants.split('.'):
        print(reactant)
        mol = Mol(smarts=reactant) 
        atom_feat = mol.get_atom_feature()
        adjacency, bond_feat = mol.get_bond_feature()
        frag_feat = mol.get_func()

        print('atom:',atom_feat)
        print('adjacency',adjacency)
        print('bond',bond_feat)
        print('frag',frag_feat)
        
    