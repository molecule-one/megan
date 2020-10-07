# utilities for using MEGAN in training and evaluation

from typing import List, Tuple, Optional, Set

import logging
import numpy as np
import pandas as pd
import torch
from src.feat import ORDERED_ATOM_OH_KEYS
from rdkit import Chem
from rdkit.Chem import rdchem, Atom, Mol, BondStereo, BondType

logger = logging.getLogger(__name__)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def set_atom_feat(a: Atom, key: str, val: int):
    if key == 'atomic_num':
        a.SetAtomicNum(val)
    elif key == 'formal_charge':
        a.SetFormalCharge(val)
    elif key == 'chiral_tag':
        a_chiral = rdchem.ChiralType.values[val]
        a.SetChiralTag(a_chiral)
    elif key == 'num_explicit_hs':
        a.SetNumExplicitHs(val)
    elif key == 'is_aromatic':
        a.SetIsAromatic(bool(val))
    return a


class RdkitCache(object):
    def __init__(self, props: dict):
        self.cache = {}
        self.props = props

    def get_atom(self, features: Tuple[int]) -> Atom:
        if features not in self.cache:
            atom = Chem.Atom(6)
            feat_i = 0
            for key in ORDERED_ATOM_OH_KEYS:
                if key in self.props['atom']:
                    if features[feat_i] < 1:
                        raise ValueError(f"Atom {key} feat value must be >= 1")
                    val = self.props['atom'][key][features[feat_i] - 1]
                    atom = set_atom_feat(atom, key, val)
                    feat_i += 1
            self.cache[features] = atom
            return atom
        return self.cache[features]

    def get_bond(self, features: Tuple[int]) -> Tuple[int, int]:
        if features not in self.cache:
            b_type = self.props['bond']['bond_type'][features[0] - 1]
            b_stereo = self.props['bond']['bond_stereo'][features[1] - 1]
            b_type = BondType.values[b_type]
            b_stereo = BondStereo.values[b_stereo]
            bond = b_type, b_stereo
            self.cache[features] = bond
            return bond
        return self.cache[features]


def mols_from_graph(rdkit_cache: RdkitCache, input_mol: Mol, input_bond_dirs: dict,
                    adj: np.ndarray, nodes: np.ndarray, changed_atoms: Set[int], only_edited: bool = False) \
        -> List[Mol]:
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms
    n_atoms = 0
    is_edited = []

    for node_i, node in enumerate(nodes[1:]):
        if node[0] != 0:
            map_num = node_i + 1
            try:
                new_a = rdkit_cache.get_atom(tuple(node))
            except ValueError as e:
                logger.warning(f'Exception while RdkitCache.get_atom: {str(e)}')
                # in case of an unknown atom type, copy atom from input mol
                new_a = input_mol.GetAtomWithIdx(map_num - 1)
            new_a.SetAtomMapNum(map_num)
            mol.AddAtom(new_a)
            n_atoms += 1
            is_edited.append(map_num in changed_atoms)

    # add bonds between adjacent atoms
    nonzero_x, nonzero_y = np.nonzero(np.triu(adj[1:, 1:, 0], k=1))

    neighbors = {i: set() for i in range(n_atoms)}

    for i in range(len(nonzero_x)):
        ix, iy = int(nonzero_x[i]), int(nonzero_y[i])
        bond = adj[ix + 1, iy + 1]
        neighbors[ix].add(iy)
        neighbors[iy].add(ix)

        b_type, b_stereo = rdkit_cache.get_bond(tuple(bond))
        bond_ind = mol.AddBond(ix, iy, order=b_type) - 1
        new_bond = None
        if b_stereo != BondStereo.STEREONONE:
            new_bond = mol.GetBondWithIdx(bond_ind)
            new_bond.SetStereo(b_stereo)

        bond_atoms = ix, iy
        if bond_atoms in input_bond_dirs:
            if new_bond is None:
                new_bond = mol.GetBondWithIdx(bond_ind)
            new_bond.SetBondDir(input_bond_dirs[bond_atoms])

    # remove compounds that have not been edited (needed for properly evaluating on some datasets, ie USPTO-MIT)
    if only_edited:
        def mark_edited(atom_i):
            is_edited[atom_i] = True
            for atom_j in neighbors[atom_i]:
                if not is_edited[atom_j]:
                    mark_edited(atom_j)

        for i in range(n_atoms):
            if is_edited[i]:
                mark_edited(i)

        for i in reversed(range(n_atoms)):
            if not is_edited[i]:
                mol.RemoveAtom(i)

    return [mol]


def get_base_action_masks(n_max_nodes: int, action_vocab: dict) -> dict:
    node_mask = torch.ones((n_max_nodes,), device=device, dtype=torch.long)

    atom_action_mask = torch.ones((*node_mask.shape, action_vocab['n_atom_actions']),
                                  dtype=torch.long, device=device)
    bond_action_mask = torch.ones((*node_mask.shape, n_max_nodes, action_vocab['n_bond_actions']),
                                  dtype=torch.long, device=device)

    # supernode (first node) can only predict "stop" action
    # (and "stop" action can be predicted only by supernode)
    # (this masking is always applied)
    bond_action_mask[0, :] = 0
    bond_action_mask[:, 0] = 0
    atom_action_mask[0] = 0
    atom_action_mask[0, action_vocab['stop_action_num']] = 1
    atom_action_mask[1:, action_vocab['stop_action_num']] = 0

    # mask out bond actions for diagonal ('self' node)
    # mask out bond actions for upper half (matrix is symmetric)
    triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int, device=device), diagonal=1)
    triu = triu.unsqueeze(-1)
    bond_action_mask *= triu

    # noinspection PyArgumentList
    return {
        'node_mask': node_mask.unsqueeze(-1),
        'atom_action_mask': atom_action_mask,
        'bond_action_mask': bond_action_mask,
        'vocab': action_vocab
    }


def generate_eval_batch(mol_graphs: List, base_action_masks: dict, reaction_types: Optional[np.ndarray] = None) -> dict:
    n_max_nodes = 0
    for adj, nodes in mol_graphs:
        n_max_nodes = max(n_max_nodes, len(nodes))

    batch_adj = torch.zeros((len(mol_graphs), n_max_nodes, n_max_nodes, mol_graphs[0][0].shape[-1]),
                            dtype=torch.long, device=device)
    node_feats = torch.zeros((len(mol_graphs), n_max_nodes, mol_graphs[0][1].shape[-1]),
                             dtype=torch.long, device=device)

    for i, (adj, nodes) in enumerate(mol_graphs):
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, dtype=torch.long, device=device)
            nodes = torch.tensor(nodes, dtype=torch.long, device=device)

        batch_adj[i, :adj.shape[0], :adj.shape[1]] = adj
        node_feats[i, :len(nodes)] = nodes

    node_mask = torch.sign(torch.max(node_feats, dim=-1)[0])
    adj_mask = torch.sign(torch.max(batch_adj, dim=-1)[0]).unsqueeze(-1)
    atom_action_mask = base_action_masks['atom_action_mask'][:n_max_nodes].unsqueeze(0).clone()
    bond_action_mask = base_action_masks['bond_action_mask'][:n_max_nodes, :n_max_nodes].unsqueeze(0)
    bond_action_mask = bond_action_mask.expand((node_mask.shape[0], -1, -1, -1)).clone()

    # only bonds between existing atoms can be edited
    atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
    bond_action_mask *= atom_exists

    # noinspection PyArgumentList
    result = {
        'node_features': node_feats,
        'node_mask': node_mask.unsqueeze(-1),
        'adj': batch_adj,
        'adj_mask': adj_mask,
        'atom_action_mask': atom_action_mask,
        'bond_action_mask': bond_action_mask
    }
    if reaction_types is not None:
        result['reaction_type'] = torch.tensor(reaction_types, dtype=torch.long, device=device) - 1  # starts at 1
        result['reaction_type'] = torch.clamp(result['reaction_type'], min=0)

    return result


def generate_batch(graph_ind: np.ndarray, metadata: pd.DataFrame, featurizer, data, action_vocab: dict) -> dict:
    sample_ind = []
    reac_n_steps = []
    n_paths = 0
    reaction_class = []

    if 'n_paths' in metadata:
        paths_per_reaction = []
        for ind in graph_ind:
            n_p = 1
            paths_per_reaction.append(n_p)
            n_paths += n_p
            for path_i in range(n_p):
                path_ind = ind + path_i
                start_ind = metadata['start_ind'][path_ind]
                n_steps = metadata['n_samples'][path_ind]
                sample_ind.append(np.arange(start_ind, start_ind + n_steps))
                reac_n_steps.append(n_steps)
                if 'class' in metadata:
                    reaction_class.append(metadata['class'][path_ind])
    else:
        paths_per_reaction = [1 for _ in range(len(graph_ind))]
        n_paths = len(graph_ind)
        for ind in graph_ind:
            start_ind = metadata['start_ind'][ind]
            n_steps = metadata['n_samples'][ind]
            sample_ind.append(np.arange(start_ind, start_ind + n_steps))
            reac_n_steps.append(n_steps)
            if 'class' in metadata:
                reaction_class.append(metadata['class'][ind])

    paths_per_reaction = torch.tensor(paths_per_reaction, dtype=torch.long, device=device)
    reac_n_steps = torch.tensor(reac_n_steps, dtype=torch.long, device=device)
    n_max_steps = max(reac_n_steps)
    sample_ind = np.concatenate(sample_ind)
    sample_data = data['sample_data'][sample_ind]

    if hasattr(sample_data, 'toarray'):
        sample_data = sample_data.toarray().astype(int)

    action_ind, atom_map1, atom_map2, n_nodes, is_hard = \
        sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], sample_data[:, 3], sample_data[:, 4]

    n_max_nodes = int(max(n_nodes))

    tensor_data = {
        'atom': data['atom'][sample_ind],
        'bond': data['bond'][sample_ind],
        'max_n_nodes': n_max_nodes
    }

    if sample_data.shape[1] > 5:
        reaction_type = sample_data[:, 5]
        tensor_data['reaction_type'] = reaction_type

    tensor_data = featurizer.to_tensor_batch(tensor_data, action_vocab)

    node_feats = torch.zeros((n_paths, n_max_steps, n_max_nodes, tensor_data['atom'].shape[-1]),
                             dtype=torch.long, device=device)

    adj = torch.zeros((n_paths, n_max_steps, n_max_nodes, n_max_nodes, tensor_data['bond'].shape[-1]),
                      dtype=torch.long, device=device)

    is_hard_matrix = torch.zeros((n_paths, n_max_steps), dtype=torch.long, device=device)
    reaction_type = torch.zeros((n_paths, n_max_steps), dtype=torch.long, device=device)

    k = 0
    for i, n_steps in enumerate(reac_n_steps):
        for j in range(n_steps):
            node_feats[i, j] = tensor_data['atom'][k]
            adj[i, j] = tensor_data['bond'][k]
            if is_hard[k]:
                is_hard_matrix[i, j] = 1
            if 'reaction_type' in tensor_data:
                reaction_type[i, j] = tensor_data['reaction_type'][k] - 1  # reaction types are numbered from 1
            k += 1

    node_mask = torch.sign(torch.max(node_feats, dim=-1)[0])
    adj_mask = torch.sign(torch.max(adj, dim=-1)[0]).unsqueeze(-1)

    atom_action_mask = torch.ones((*node_mask.shape, action_vocab['n_atom_actions']),
                                  dtype=torch.float, device=device)
    bond_action_mask = torch.ones((*node_mask.shape, n_max_nodes, action_vocab['n_bond_actions']),
                                  dtype=torch.float, device=device)

    # supernode (first node) can only predict "stop" action
    # (and "stop" action can be predicted only by supernode)
    # (this masking is always applied)
    bond_action_mask[:, :, 0, :] = 0
    bond_action_mask[:, :, :, 0] = 0
    atom_action_mask[:, :, 0] = 0
    atom_action_mask[:, :, 0, action_vocab['stop_action_num']] = 1
    atom_action_mask[:, :, 1:, action_vocab['stop_action_num']] = 0

    # mask out bond actions for diagonal ('self' node)
    # mask out bond actions for upper half (matrix is symmetric)
    triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int, device=device), diagonal=1)
    triu = triu.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    bond_action_mask *= triu

    # only bonds between existing atoms can be edited
    atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
    bond_action_mask *= atom_exists

    target = torch.zeros((n_paths, n_max_steps, n_max_nodes + 1, n_max_nodes,
                          action_vocab['n_target_actions']), dtype=torch.float, device=device)

    node_mask = node_mask.unsqueeze(-1)

    result = {
        'node_features': node_feats,
        'node_mask': node_mask,
        'adj': adj,
        'adj_mask': adj_mask,
        'target': target,
        'atom_action_mask': atom_action_mask,
        'bond_action_mask': bond_action_mask,
        'n_steps': reac_n_steps,
        'is_hard': is_hard_matrix,
        'n_paths': paths_per_reaction,
    }
    if 'reaction_type' in tensor_data:
        result['reaction_type'] = reaction_type

    k = 0
    for i, n_steps in enumerate(reac_n_steps):
        for j in range(n_steps):
            action_num, a1, a2 = action_ind[k], atom_map1[k], atom_map2[k]
            if a1 < a2:
                a2, a1 = a1, a2
            action_num = action_vocab['atom_action_num'][action_num] if a2 == -1 \
                else action_vocab['bond_action_num'][action_num]
            target[i, j, a2, a1, action_num] = 1
            k += 1
        for j in range(n_steps, n_max_steps):
            bond_action_mask[i, j] = 0
            atom_action_mask[i, j] = 0

    # noinspection PyArgumentList
    target = target.reshape(n_paths, n_max_steps, -1)
    result['target'] = target

    return result
