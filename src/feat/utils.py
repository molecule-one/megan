"""
Utility functions for MEGAN featurization
"""
from typing import Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import numpy as np

from src.feat.graph_features import try_get_atom_feature
from src.feat import ATOM_EDIT_TUPLE_KEYS


def fix_incomplete_mappings(sub_mol: Mol, prod_mol: Mol) -> Tuple[Mol, Mol]:
    max_map = max(a.GetAtomMapNum() for a in sub_mol.GetAtoms())
    max_map = max(max(a.GetAtomMapNum() for a in prod_mol.GetAtoms()), max_map)

    for mol in (sub_mol, prod_mol):
        for a in mol.GetAtoms():
            map_num = a.GetAtomMapNum()
            if map_num is None or map_num < 1:
                max_map += 1
                a.SetAtomMapNum(max_map)
    return sub_mol, prod_mol


def add_map_numbers(mol: Mol) -> Mol:
    # converting to smiles to mol and again to smiles makes atom order canonical
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    map_nums = np.arange(mol.GetNumAtoms()) + 1
    np.random.shuffle(map_nums)

    for i, a in enumerate(mol.GetAtoms()):
        a.SetAtomMapNum(int(map_nums[i]))
    return mol


def reac_to_canonical(sub_mol, prod_mol): # converting to smiles to mol and again to smiles makes atom order canonical
    sub_mol = Chem.MolFromSmiles(Chem.MolToSmiles(sub_mol))
    prod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol))

    # in RdKit chirality can be marked different depending on order of atoms in molecule list
    # here we remap atoms so the map order is consistent with atom list order

    map2map = {}
    for i, a in enumerate(prod_mol.GetAtoms()):
        map2map[a.GetAtomMapNum()] = i + 1
        a.SetAtomMapNum(i + 1)

    max_map = max(map2map.values())
    for i, a in enumerate(sub_mol.GetAtoms()):
        m = a.GetAtomMapNum()
        if m in map2map:
            a.SetAtomMapNum(map2map[m])
        else:
            max_map += 1
            a.SetAtomMapNum(max_map)

    return sub_mol, prod_mol


def get_bond_tuple(bond) -> Tuple[int, int, int, int]:
    a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
    bt = int(bond.GetBondType())
    st = int(bond.GetStereo())
    if a1 > a2:
        a1, a2 = a2, a1
    return a1, a2, bt, st


def atom_to_edit_tuple(atom) -> Tuple:
    feat = [try_get_atom_feature(atom, key) for key in ATOM_EDIT_TUPLE_KEYS]
    return tuple(feat)


# rdkit has a problem with implicit hs. By default there are only explicit hs.
# This is a hack to fix this error
def fix_explicit_hs(mol: Mol) -> Mol:
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)

    mol = Chem.AddHs(mol, explicitOnly=True)
    mol = Chem.RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol


def get_atom_ind(mol: Mol, atom_map: int) -> int:
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomMapNum() == atom_map:
            return i
    raise ValueError(f'No atom with map number: {atom_map}')
