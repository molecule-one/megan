"""
Special actions for adding some kinds of rings
"""
from typing import List
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.feat.utils import atom_to_edit_tuple


def add_benzene_ring(mol: Mol, start_atom_ind: int, ring_atom_maps: List[int]):
    new_atom_ind = []

    map2i = dict((a.GetAtomMapNum(), i) for i, a in enumerate(mol.GetAtoms()))

    start_atom = mol.GetAtomWithIdx(start_atom_ind)
    start_atom.SetBoolProp('is_edited', True)
    start_atom.SetIsAromatic(True)
    start_atom_map = start_atom.GetAtomMapNum()

    if start_atom.HasProp('in_reactant'):
        in_reactant = start_atom.GetBoolProp('in_reactant')
    else:
        in_reactant = False

    if start_atom.HasProp('mol_id'):
        mol_id = start_atom.GetIntProp('mol_id')
    else:
        mol_id = 1

    for atom_map in ring_atom_maps:
        if atom_map != start_atom_map:
            if atom_map in map2i:
                new_atom_ind.append(map2i[atom_map])
            else:
                num_atoms = mol.GetNumAtoms()
                new_a = Chem.Atom(6)  # benzene has only carbon atoms
                new_a.SetAtomMapNum(atom_map)
                new_a.SetIsAromatic(True)
                new_a.SetBoolProp('is_edited', True)
                new_a.SetBoolProp('in_reactant', in_reactant)
                new_a.SetIntProp('mol_id', mol_id)
                mol.AddAtom(new_a)
                new_atom_ind.append(num_atoms)
        else:
            new_atom_ind.append(start_atom_ind)

    for i in range(len(new_atom_ind) - 1):
        bond = mol.GetBondBetweenAtoms(new_atom_ind[i], new_atom_ind[i + 1])
        if bond is None:
            bond_idx = mol.AddBond(new_atom_ind[i], new_atom_ind[i + 1], order=Chem.rdchem.BondType.AROMATIC) - 1
            bond = mol.GetBondWithIdx(bond_idx)
        bond.SetBoolProp('is_edited', True)

    bond = mol.GetBondBetweenAtoms(new_atom_ind[0], new_atom_ind[-1])
    if bond is None:
        bond_idx = mol.AddBond(new_atom_ind[0], new_atom_ind[-1], order=Chem.rdchem.BondType.AROMATIC) - 1
        bond = mol.GetBondWithIdx(bond_idx)
    bond.SetBoolProp('is_edited', True)

    return mol


def find_rings(mol: Mol) -> List[List[int]]:
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    i2map = dict((i, a.GetAtomMapNum()) for i, a in enumerate(mol.GetAtoms()))
    rings_mapped = []

    for ring in rings:
        rings_mapped.append([i2map[i] for i in ring])

    return rings_mapped


def is_benzene_ring(ring_atoms):
    return len(ring_atoms) == 6 and all(atom_to_edit_tuple(a) == (0, 0, 0, 1) and a.GetAtomicNum() == 6 for a in ring_atoms)


def find_added_benzene_rings(source_mol: Mol, target_mol: Mol) -> List[List[int]]:
    """
    Find benzene rings that were added in the process of reaction generation
    """
    target_rings = find_rings(target_mol)

    map2atom = dict((a.GetAtomMapNum(), a) for i, a in enumerate(target_mol.GetAtoms()))
    source_atoms = set(a.GetAtomMapNum() for a in source_mol.GetAtoms())

    added_benzene_rings = []
    for ring in target_rings:
        if all(m not in source_atoms for m in ring) and is_benzene_ring([map2atom[m] for m in ring]):
            added_benzene_rings.append(ring)

    return added_benzene_rings
