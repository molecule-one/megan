"""
Definitions of basic 'edits' (Actions) to transform a target into substrates
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, BondType, ChiralType, BondStereo, RWMol
from rdkit.Chem.rdchem import GetPeriodicTable

from src.feat.graph_features import get_atom_features
from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ATOM_EDIT_TUPLE_KEYS
from src.feat.ring_actions import add_benzene_ring
from src.feat.utils import get_atom_ind

PERIODIC_TABLE = GetPeriodicTable()


class ReactionAction(metaclass=ABCMeta):
    def __init__(self, atom_map1: int, atom_map2: int, action_vocab: dict, is_hard: bool = False):
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.is_hard = is_hard
        self.action_vocab = action_vocab
        self.prop2oh = action_vocab['prop2oh']

    @abstractmethod
    def get_tuple(self) -> Tuple[str, ...]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def apply(self, mol: RWMol) -> RWMol:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError('Abstract method')


feat_key_to_str = {
    'atomic_num': 'Atomic Number',
    'formal_charge': 'Formal Charge',
    'is_aromatic': 'Is Aromatic',
    'chiral_tag': 'Chiral Type',
    'num_explicit_hs': 'Num. Explicit Hs',
    'bond_type': 'Bond Type',
    'bond_stereo': 'Bond Stereo',
}


def feat_val_to_str(feat_key, feat_val, with_key=True):
    if feat_key == 'atomic_num':
        str_val = PERIODIC_TABLE.GetElementSymbol(feat_val)
    elif feat_key == 'formal_charge':
        if with_key:
            str_val = str(feat_val)
            if feat_val > 0:
                str_val = '+' + str_val
        else:
            str_val = ''
            if feat_val == -1:
                str_val = '-'
            elif feat_val == 1:
                str_val = '+'
            elif feat_val > 1:
                str_val = f'{feat_val}+'
            elif feat_val < -1:
                str_val = f'{feat_val}-'
    elif feat_key == 'is_aromatic':
        str_val = 'Yes' if feat_val == 1 else 'No'
    elif feat_key == 'chiral_tag':
        if feat_val == ChiralType.CHI_UNSPECIFIED:
            str_val = 'None'
        elif feat_val == ChiralType.CHI_TETRAHEDRAL_CW:
            str_val = 'CW'
        elif feat_val == ChiralType.CHI_TETRAHEDRAL_CCW:
            str_val = 'CCW'
        elif feat_val == ChiralType.OTHER:
            str_val = 'Other'
        else:
            str_val = str(feat_val)
    elif feat_key == 'num_explicit_hs':
        if with_key:
            return str(feat_val)
        else:
            str_val = ''
            if feat_val == 1:
                str_val = f'H'
            elif feat_val > 1:
                str_val = f'{feat_val}H'
    elif feat_key == 'bond_type':
        str_val = str(BondType.values[feat_val]).capitalize()
    elif feat_key == 'bond_stereo':
        str_val = str(BondStereo.values[feat_val])[6:].capitalize()
    else:
        str_val = str(feat_val)

    if with_key:
        return f'{feat_key_to_str[feat_key]}={str_val}'
    return str_val


def atom_to_str(atomic_num, formal_charge, num_explicit_hs):
    symbol = feat_val_to_str('atomic_num', atomic_num, with_key=False)
    charge = feat_val_to_str('formal_charge', formal_charge, with_key=False)
    hs = feat_val_to_str('num_explicit_hs', num_explicit_hs, with_key=False)
    return symbol + hs + charge


class AtomEditAction(ReactionAction):
    def __init__(self, atom_map1: int, formal_charge: int, chiral_tag: int,
                 num_explicit_hs: int, is_aromatic: int, action_vocab: dict,
                 is_hard: bool = False):
        super(AtomEditAction, self).__init__(atom_map1, -1, action_vocab, is_hard)
        self.formal_charge = formal_charge
        self.chiral_tag = chiral_tag
        self.num_explicit_hs = num_explicit_hs
        self.is_aromatic = is_aromatic
        self.feat_keys = ATOM_EDIT_TUPLE_KEYS + ['is_edited']

    @property
    def feat_vals(self) -> Tuple[int, int, int, int, int]:
        return self.formal_charge, self.chiral_tag, self.num_explicit_hs, self.is_aromatic, 1

    def get_tuple(self) -> Tuple[str, Tuple[int, int, int, int]]:
        return 'change_atom', self.feat_vals[:-1]

    def apply(self, mol: RWMol) -> RWMol:
        atom_ind = get_atom_ind(mol, self.atom_map1)
        atom = mol.GetAtomWithIdx(atom_ind)

        atom.SetFormalCharge(self.formal_charge)
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]
        atom.SetChiralTag(a_chiral)
        atom.SetNumExplicitHs(self.num_explicit_hs)
        atom.SetIsAromatic(self.is_aromatic)
        atom.SetBoolProp('is_edited', True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nodes = nodes.copy()
        for key, val in zip(self.feat_keys, self.feat_vals):
            ind = self.action_vocab['atom_feat_ind'].get(key, -1)
            if ind != -1:
                ind = self.action_vocab['atom_feat_ind'][key]
                nodes[self.atom_map1, ind] = self.prop2oh['atom'][key].get(val, 0)
        return adj, nodes

    def __str__(self):
        feat_vals = ', '.join([feat_val_to_str(key, val) for val, key in
                               zip(self.feat_vals, self.feat_keys) if key != 'is_edited'])
        return f'Edit Atom {self.atom_map1}: {feat_vals}'


class AddAtomAction(ReactionAction):
    def __init__(self, atom_map1: int, atom_map2: int,
                 bond_type: int, bond_stereo: int,
                 atomic_num: int, formal_charge: int, chiral_tag: int,
                 num_explicit_hs: int, is_aromatic: int, action_vocab: dict,
                 is_hard: bool = False, detach: bool = False):
        super(AddAtomAction, self).__init__(atom_map1, atom_map2, action_vocab, is_hard)
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo
        self.atomic_num = atomic_num
        self.formal_charge = formal_charge
        self.chiral_tag = chiral_tag
        self.num_explicit_hs = num_explicit_hs
        self.is_aromatic = is_aromatic
        self.detach = detach

        self.new_a = self._gen_new_atom()
        self.new_atom_features = get_atom_features(self.new_a, ORDERED_ATOM_OH_KEYS, atom_prop2oh=self.prop2oh['atom'])

        # new atom has 1 neighbour when its created
        self.degree_ind = self.action_vocab['atom_feat_ind'].get('degree', -1)
        if self.degree_ind != -1:
            self.new_atom_features[self.degree_ind] = self.prop2oh['atom']['degree'][1]

        self.mol_id_ind = self.action_vocab['atom_feat_ind'].get('mol_id', -1)
        self.is_reactant_ind = self.action_vocab['atom_feat_ind'].get('is_reactant', -1)

    @property
    def atom_feat_vals(self):
        return self.atomic_num, self.formal_charge, self.chiral_tag, self.num_explicit_hs, self.is_aromatic

    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int]]]:
        return 'add_atom', ((self.bond_type, self.bond_stereo), self.atom_feat_vals)

    def _gen_new_atom(self):
        new_a = Chem.Atom(self.atomic_num)
        new_a.SetFormalCharge(self.formal_charge)
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]
        new_a.SetChiralTag(a_chiral)
        new_a.SetNumExplicitHs(self.num_explicit_hs)
        new_a.SetIsAromatic(self.is_aromatic)
        new_a.SetAtomMapNum(self.atom_map2)
        new_a.SetBoolProp('is_edited', True)
        return new_a

    def _get_bond_features(self):
        return [self.bond_type, self.bond_stereo, 1]

    def apply(self, mol: RWMol) -> RWMol:
        num_atoms = mol.GetNumAtoms()
        if self.detach:
            for i, a in enumerate(mol.GetAtoms()):
                m = a.GetAtomMapNum()
                if m == self.atom_map2:
                    for bond in a.GetBonds():
                        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    mol.RemoveAtom(i)
                    num_atoms -= 1
                    break

        atom_ind = get_atom_ind(mol, self.atom_map1)
        b_type = rdchem.BondType.values[self.bond_type]
        b_stereo = rdchem.BondStereo.values[self.bond_stereo]

        old_atom = mol.GetAtomWithIdx(atom_ind)
        if old_atom.HasProp('in_reactant'):
            self.new_a.SetBoolProp('in_reactant', old_atom.GetBoolProp('in_reactant'))
        if old_atom.HasProp('mol_id'):
            self.new_a.SetIntProp('mol_id', old_atom.GetIntProp('mol_id'))

        mol.AddAtom(self.new_a)
        new_atom_ind = num_atoms

        bond_ind = mol.AddBond(atom_ind, new_atom_ind, order=b_type) - 1
        new_bond = mol.GetBondWithIdx(bond_ind)
        new_bond.SetStereo(b_stereo)
        new_bond.SetBoolProp('is_edited', True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_num = max(len(adj), self.atom_map2 + 1)
        new_adj = np.full((max_num, max_num, adj.shape[2]), fill_value=0, dtype=int)
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_num, nodes.shape[1]), fill_value=0, dtype=int)
        new_nodes[:nodes.shape[0]] = nodes
        new_nodes[self.atom_map2] = self.new_atom_features

        if self.detach:
            for i in range(1, len(new_nodes)):
                if self.degree_ind != -1 and i != self.atom_map2 and new_adj[self.atom_map2, i, 0] != 0:
                    new_nodes[i, self.degree_ind] -= 1
            new_adj[self.atom_map2, :] = 0
            new_adj[:, self.atom_map2] = 0

        # copy "mol_id" and "is_reactant" to new atom from the old neighboring atom
        if self.mol_id_ind != -1:
            new_nodes[self.atom_map2, self.mol_id_ind] = new_nodes[self.atom_map1, self.mol_id_ind]
        if self.is_reactant_ind != -1:
            new_nodes[self.atom_map2, self.is_reactant_ind] = new_nodes[self.atom_map1, self.is_reactant_ind]

        # update 'degree' feature of old atom (+= 1)
        if self.degree_ind != -1:
            new_nodes[self.atom_map1, self.degree_ind] += 1

        bond_features = [self.prop2oh['bond'][key].get(val, 0) for key, val in
                         zip(ORDERED_BOND_OH_KEYS, self._get_bond_features())]

        new_adj[self.atom_map1, self.atom_map2] = new_adj[self.atom_map2, self.atom_map1] = bond_features

        new_adj[0, self.atom_map2, 0] = new_adj[self.atom_map2, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']
        new_adj[self.atom_map2, self.atom_map2, 0] = self.prop2oh['bond']['bond_type']['self']

        return new_adj, new_nodes

    def __str__(self):
        new_atom_str = atom_to_str(self.atomic_num, self.formal_charge, self.num_explicit_hs)
        key = 'Detach' if self.detach else 'Add'
        return f'{key} {new_atom_str}:{self.atom_map2} to atom {self.atom_map1} ' \
               f'({feat_val_to_str("bond_type", self.bond_type)})'


class AddRingAction(ReactionAction):
    def __init__(self, atom_map1: int, new_atoms_map_nums: List[int], ring_key: str,
                 action_vocab: dict, is_hard: bool = False):
        super(AddRingAction, self).__init__(atom_map1, -1, action_vocab, is_hard)
        self.ring_key = ring_key

        # order new atom map nums so map num of the existing atom is first
        map_ind = new_atoms_map_nums.index(self.atom_map1)
        self.new_atoms_map_nums = [self.atom_map1] + new_atoms_map_nums[map_ind + 1:] + new_atoms_map_nums[:map_ind]

        new_a = Chem.Atom(6)
        new_a.SetIsAromatic(True)
        new_a.SetBoolProp('is_edited', True)
        self.new_atom_features = get_atom_features(new_a, ORDERED_ATOM_OH_KEYS, atom_prop2oh=self.prop2oh['atom'])

        b_type = Chem.rdchem.BondType.AROMATIC
        self.new_bond_features = [self.prop2oh['bond'][key][val] for key, val in
                                  (('bond_type', b_type), ('bond_stereo', 0), ('is_edited', 1))]

    def get_tuple(self) -> Tuple[str, str]:
        return 'add_ring', self.ring_key

    def apply(self, mol: RWMol) -> RWMol:
        atom_ind = get_atom_ind(mol, self.atom_map1)
        if self.ring_key == 'benzene':
            mol = add_benzene_ring(mol, start_atom_ind=atom_ind, ring_atom_maps=self.new_atoms_map_nums)
        else:
            raise ValueError(f'No such ring type: {self.ring_key}')
        return mol

    def __str__(self):
        return f'Add {self.ring_key} ring to atom {self.atom_map1}'

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.ring_key != 'benzene':
            raise ValueError(f'No such ring type: {self.ring_key}')
        max_map_num = max(max(self.new_atoms_map_nums) + 1, adj.shape[0])

        new_adj = np.full((max_map_num, max_map_num, adj.shape[2]), fill_value=0, dtype=int)
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_map_num, nodes.shape[1]), fill_value=0, dtype=int)
        new_nodes[:nodes.shape[0]] = nodes

        degree_ind = self.action_vocab['atom_feat_ind'].get('degree', -1)
        mol_id_ind = self.action_vocab['atom_feat_ind'].get('mol_id', -1)
        is_reactant_ind = self.action_vocab['atom_feat_ind'].get('is_reactant', -1)

        if mol_id_ind != -1:
            self.new_atom_features[mol_id_ind] = nodes[self.atom_map1, mol_id_ind]
        if is_reactant_ind != -1:
            self.new_atom_features[is_reactant_ind] = nodes[self.atom_map1, is_reactant_ind]

        for i, map_num in enumerate(self.new_atoms_map_nums):
            if degree_ind != -1:
                old_degree = new_nodes[map_num, degree_ind]
            else:
                old_degree = -1
            new_nodes[map_num] = self.new_atom_features

            # starting node has degree increased by 2 (it "closes" the ring)
            if degree_ind != -1:
                if map_num == self.atom_map1:
                    new_nodes[map_num, degree_ind] = old_degree + 2
                # all other nodes have degree 2
                else:
                    new_nodes[map_num, degree_ind] = self.prop2oh['atom']['degree'][2]

            new_adj[map_num, map_num, 0] = self.prop2oh['bond']['bond_type']['self']
            new_adj[0, map_num, 0] = new_adj[map_num, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']

            if i > 0:
                prev_map_num = self.new_atoms_map_nums[i - 1]
                new_adj[prev_map_num, map_num] = new_adj[map_num, prev_map_num] = self.new_bond_features

        # close the ring (connected first atom to the last)
        map_num2 = self.new_atoms_map_nums[-1]
        new_adj[self.atom_map1, map_num2] = new_adj[map_num2, self.atom_map1] = self.new_bond_features

        return new_adj, new_nodes


class BondEditAction(ReactionAction):
    def __init__(self, atom_map1: int, atom_map2: int,
                 bond_type: Optional[int], bond_stereo: Optional[int],
                 action_vocab: dict, is_hard: bool = False):
        super(BondEditAction, self).__init__(atom_map1, atom_map2, action_vocab, is_hard)
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo
        self.bond_feat_keys = ['bond_type', 'bond_stereo', 'is_edited']
        self.is_aromatic_val = self.prop2oh['atom']['is_aromatic'][1]

    def get_tuple(self) -> Tuple[str, Tuple[Optional[int], Optional[int]]]:
        return 'change_bond', (self.bond_type, self.bond_stereo)

    def _get_bond_features(self):
        return [self.bond_type, self.bond_stereo, 1]

    def apply(self, mol: RWMol) -> RWMol:
        atom1 = get_atom_ind(mol, self.atom_map1)
        atom2 = get_atom_ind(mol, self.atom_map2)

        if self.bond_type is None:  # delete bond
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                mol.RemoveBond(atom1, atom2)
        else:
            b_type = rdchem.BondType.values[self.bond_type]
            b_stereo = rdchem.BondStereo.values[self.bond_stereo]

            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is None:  # add new bond
                bond_ind = mol.AddBond(atom1, atom2, order=b_type) - 1
                bond = mol.GetBondWithIdx(bond_ind)
            else:  # change an existing bond
                bond.SetBondType(b_type)
            bond.SetStereo(b_stereo)
            bond.SetBoolProp('is_edited', True)

            if b_type == BondType.AROMATIC:
                bond.SetIsAromatic(True)
                mol.GetAtomWithIdx(atom1).SetIsAromatic(True)
                mol.GetAtomWithIdx(atom2).SetIsAromatic(True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adj = adj.copy()
        nodes = nodes.copy()

        degree_ind = self.action_vocab['atom_feat_ind'].get('degree', -1)
        is_aromatic_ind = self.action_vocab['atom_feat_ind'].get('is_aromatic', -1)

        if self.bond_type is None:
            bond_features = [0, 0, 0]

            # for a deleted bond, decrease degree of nodes by 1
            if degree_ind != -1 and adj[self.atom_map1, self.atom_map2, 0] != 0:
                nodes[self.atom_map1, degree_ind] -= 1
                nodes[self.atom_map2, degree_ind] -= 1

        else:
            bond_features = [self.prop2oh['bond'][key].get(val, 0) for key, val in zip(self.bond_feat_keys,
                                                                                       self._get_bond_features())]
            # for a new bond, increase degree of nodes by 1
            if degree_ind != -1 and adj[self.atom_map1, self.atom_map2, 0] == 0:
                nodes[self.atom_map1, degree_ind] += 1
                nodes[self.atom_map2, degree_ind] += 1

        if is_aromatic_ind != -1 and self.bond_type == 12:  # aromatic bond
            nodes[self.atom_map1, is_aromatic_ind] = self.is_aromatic_val
            nodes[self.atom_map2, is_aromatic_ind] = self.is_aromatic_val

        adj[self.atom_map1, self.atom_map2] = adj[self.atom_map2, self.atom_map1] = bond_features

        return adj, nodes

    def __str__(self):
        if self.bond_type is None:
            return f'Delete bond {self.atom_map1, self.atom_map2}'
        bond_type = f'{feat_val_to_str("bond_type", self.bond_type)}'
        bond_stereo = f'{feat_val_to_str("bond_stereo", self.bond_stereo)}'
        return f'Edit bond {self.atom_map1, self.atom_map2}: {bond_type}, {bond_stereo}'


class StopAction(ReactionAction):
    def __init__(self, action_vocab: dict):
        super(StopAction, self).__init__(0, -1, action_vocab=action_vocab, is_hard=False)

    def get_tuple(self) -> Tuple[str]:
        return 'stop',

    def apply(self, mol: RWMol) -> RWMol:
        return mol  # do nothing (stop generation)

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return adj, nodes  # do nothing (stop generation)

    def __str__(self):
        return 'Stop'
