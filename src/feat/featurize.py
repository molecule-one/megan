"""
Main functions for featurization for MEGAN reaction generator
"""
import itertools
import logging
import os
import random
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Mol
from scipy import sparse

from src.feat.mol_graph import get_graph
from src.feat.reaction_actions import ReactionAction, AddRingAction, AddAtomAction, BondEditAction, AtomEditAction, \
    StopAction
from src.feat.ring_actions import find_added_benzene_rings
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, \
    fix_explicit_hs
from src.utils import renumber_atoms_for_mapping, mark_reactants

logger = logging.getLogger(__name__)


class ReactionSampleGenerator(object):
    def __init__(self, source_mol: RWMol, target_mol: Mol, action_vocab: dict,
                 forward: bool = False, action_order: str = 'dfs'):
        self.source_mol = source_mol
        self.target_mol = target_mol

        self.randomize_action_types = 'random' in action_order
        self.randomize_map_atom_order = action_order == 'random' or 'randat' in action_order
        self.randomize_next_atom = action_order == 'random'

        self.action_order = action_order

        self.atoms_stack = []
        if 'bfs' in self.action_order:
            for a in target_mol.GetAtoms():
                self.atoms_stack.append(a.GetAtomMapNum())
            self.atoms_stack = list(sorted(self.atoms_stack))

        mark_reactants(source_mol, target_mol)

        self.edited_atoms = set()
        self.forward = forward
        self.action_vocab = action_vocab
        self.prop_dict = action_vocab['prop2oh']

        self.added_rings = {
            'benzene': find_added_benzene_rings(source_mol=source_mol, target_mol=target_mol)
        }
        self.current_step = 0
        self.current_mol_graph = get_graph(self.source_mol, ravel=False, to_array=True,
                                           atom_prop2oh=self.prop_dict['atom'], bond_prop2oh=self.prop_dict['bond'])

    def generate_gen_action(self) -> ReactionAction:
        atoms_only_in_target = {}
        source_atoms = {}
        target_atoms = {}
        target_atomic_nums = {}

        source_bonds = {}
        target_bonds = {}

        map2target_atom = {}

        changed_atoms = {}
        changed_bonds = {}
        added_bonds = {}
        deleted_bonds = {}
        new_rings = {}
        new_atoms = {}

        def find_add_new_atom_actions(source_a: int, new_a: int, bond_type: Tuple[int, int]):
            if new_a in atoms_only_in_target and source_a not in atoms_only_in_target:
                ring_added = False
                for ring_key, all_ring_atom_maps in self.added_rings.items():
                    for ring_atom_maps in all_ring_atom_maps:
                        if source_a in ring_atom_maps and \
                                all(m == source_a or m in atoms_only_in_target for m in ring_atom_maps):
                            new_rings[source_a] = AddRingAction(
                                atom_map1=source_a,
                                new_atoms_map_nums=ring_atom_maps,
                                ring_key=ring_key,
                                action_vocab=self.action_vocab,
                                is_hard=source_a not in self.edited_atoms)
                            ring_added = True
                            break
                    if ring_added:
                        break
                if not ring_added:
                    atomic_num = target_atomic_nums[new_a]
                    new_atoms[new_a, source_a] = AddAtomAction(source_a, new_a,
                                                               *bond_type, atomic_num, *target_atoms[new_a],
                                                               action_vocab=self.action_vocab,
                                                               is_hard=source_a not in self.edited_atoms)

        def find_new_bonds(a2: int, a1: int):
            if a1 in source_atoms and a2 in source_atoms and (a1, a2) not in source_bonds:
                added_bonds[a2, a1] = \
                    BondEditAction(a1, a2, *target_bonds[(a1, a2)], action_vocab=self.action_vocab,
                                   is_hard=a1 not in self.edited_atoms and a2 not in self.edited_atoms)

        for i, a in enumerate(self.source_mol.GetAtoms()):
            source_atoms[a.GetAtomMapNum()] = atom_to_edit_tuple(a)

        for a in self.target_mol.GetAtoms():
            am = a.GetAtomMapNum()
            at = atom_to_edit_tuple(a)
            if am not in source_atoms:
                atoms_only_in_target[am] = a
            elif source_atoms[am] != at:
                changed_atoms[am] = AtomEditAction(am, *at, action_vocab=self.action_vocab,
                                                   is_hard=am not in self.edited_atoms)
            target_atoms[am] = at
            target_atomic_nums[am] = a.GetAtomicNum()
            map2target_atom[am] = a

        for bond in self.source_mol.GetBonds():
            bond_tuple = get_bond_tuple(bond)
            source_bonds[(bond_tuple[0], bond_tuple[1])] = bond_tuple[2:]

        for bond in self.target_mol.GetBonds():
            bond_tuple = get_bond_tuple(bond)
            a1, a2, bond_type = bond_tuple[0], bond_tuple[1], bond_tuple[2:]
            target_bonds[(a1, a2)] = bond_type

            find_add_new_atom_actions(a1, a2, bond_type)
            find_add_new_atom_actions(a2, a1, bond_type)
            find_new_bonds(a2, a1)

        for bond_atoms, bond_type in source_bonds.items():
            if bond_atoms not in target_bonds:
                if bond_atoms[0] in target_atoms or bond_atoms[1] in target_atoms:
                    # find deleted bonds
                    deleted_bonds[(bond_atoms[1], bond_atoms[0])] = \
                        BondEditAction(bond_atoms[0], bond_atoms[1], None, None,
                                       action_vocab=self.action_vocab,
                                       is_hard=bond_atoms[0] not in self.edited_atoms and
                                               bond_atoms[1] not in self.edited_atoms)

            elif target_bonds[bond_atoms] != bond_type:
                # find edited bonds
                changed_bonds[(bond_atoms[1], bond_atoms[0])] = \
                    BondEditAction(bond_atoms[0], bond_atoms[1], *target_bonds[bond_atoms],
                                   action_vocab=self.action_vocab,
                                   is_hard=bond_atoms[0] not in self.edited_atoms and
                                           bond_atoms[1] not in self.edited_atoms)

        # for forward synthesis, bond addition has the highest priority regardless of atoms
        # for retrosynthesis, bond deletion has the highest priority regardless of atoms
        if self.forward:
            action_type_priorities = [
                ('double', added_bonds),
                ('double', deleted_bonds),
                ('double', changed_bonds),
                ('single', changed_atoms),
                ('single', new_rings),
                ('double', new_atoms)
            ]
        else:
            action_type_priorities = [
                ('double', deleted_bonds),
                ('double', added_bonds),
                ('double', changed_bonds),
                ('single', changed_atoms),
                ('single', new_rings),
                ('double', new_atoms)
            ]

        if self.randomize_action_types:
            random.shuffle(action_type_priorities)

        target_atom_keys = list(target_atoms.keys())

        if self.randomize_map_atom_order:
            random.shuffle(target_atom_keys)
        else:
            target_atom_keys = list(sorted(target_atom_keys))

        if self.randomize_next_atom:
            atom_maps1 = target_atom_keys
            atom_maps2 = target_atom_keys
        elif 'bfs' in self.action_order:
            atom_maps1 = self.atoms_stack
            atom_maps2 = self.atoms_stack
        else:  # dfs
            atom_maps1 = reversed(self.atoms_stack)
            atom_maps2 = itertools.chain(reversed(self.atoms_stack), target_atom_keys)

        for atom_map1 in atom_maps1:
            for action_type, actions_dict in action_type_priorities:
                if action_type == 'double':
                    for atom_map2 in atom_maps2:
                        if (atom_map1, atom_map2) in actions_dict:
                            return actions_dict[(atom_map1, atom_map2)]
                        elif (atom_map2, atom_map1) in actions_dict:
                            return actions_dict[(atom_map2, atom_map1)]
                elif atom_map1 in actions_dict:
                    return actions_dict[atom_map1]

        # if no actions found in atoms stack, go with action priorities
        for action_type, actions_dict in action_type_priorities:
            if len(actions_dict) == 0:
                continue
            action_dict_keys = list(actions_dict.keys())

            if self.randomize_map_atom_order:
                atom_maps = random.choice(action_dict_keys)
            else:
                action_dict_keys = list(sorted(action_dict_keys))
                atom_maps = action_dict_keys[0]

            return actions_dict[atom_maps]

        return StopAction(action_vocab=self.action_vocab)

    def gen_training_sample(self) -> dict:
        self.source_mol.UpdatePropertyCache(strict=False)

        # generate action
        reaction_action = self.generate_gen_action()

        training_sample = {
            'action_tuple': reaction_action.get_tuple(),
            'action_str': str(reaction_action)
        }

        adj, nodes = get_graph(self.source_mol, atom_prop2oh=self.prop_dict['atom'],
                               bond_prop2oh=self.prop_dict['bond'])

        training_sample['adj'] = adj
        training_sample['nodes'] = nodes
        training_sample['step'] = self.current_step
        training_sample['is_hard'] = reaction_action.is_hard
        training_sample['atom_map1'] = reaction_action.atom_map1

        atom_map1 = reaction_action.atom_map1
        atom_map2 = reaction_action.atom_map2
        if atom_map1 > 0:
            self.atoms_stack.append(atom_map1)
            self.edited_atoms.add(atom_map1)
        if atom_map2 > 0:
            self.atoms_stack.append(atom_map2)
            self.edited_atoms.add(atom_map2)

        if not isinstance(reaction_action, BondEditAction):
            atom_map2 = -1
        training_sample['atom_map2'] = atom_map2

        # execute action
        self.source_mol = reaction_action.apply(self.source_mol)
        self.current_mol_graph = reaction_action.graph_apply(*self.current_mol_graph)

        self.current_step += 1
        return training_sample


def gen_training_samples(target_mol: Mol, source_mol: Mol, n_max_steps: int,
                         action_vocab: dict, forward: bool = False, action_order: str = 'dfs') -> Tuple[List, str]:
    training_samples = []
    final_smi = ''

    reaction_state = ReactionSampleGenerator(
        source_mol=Chem.rdchem.RWMol(source_mol),
        target_mol=target_mol,
        forward=forward,
        action_vocab=action_vocab,
        action_order=action_order,
    )

    for i in range(n_max_steps):
        sample = reaction_state.gen_training_sample()
        training_samples.append(sample)

        if sample['action_tuple'][0] == 'stop':
            if reaction_state.source_mol is None:
                logger.warning(f'"None" mol after {i + 1} steps')
                return [], ''
            final_smi = Chem.MolToSmiles(reaction_state.source_mol)
            break

        if i >= n_max_steps - 1:
            return [], ''

    return training_samples, final_smi


def featurize_parallel(params) -> int:
    thread_num, samples_len, data_inds, data_x, max_n_nodes, feat_loop, \
    n_max_steps, is_train, rt_given, forward, action_order, \
    action_vocab, chunk_save_path = params

    n_reactions = len(data_x['substrates'])
    k = 15000  # to save RAM used by python lists, create sparse matrices every k reactions

    nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes))
    adj_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes ** 2))

    adj_vals = [], [], []  # vals, rows, cols
    nodes_vals = [], [], []  # vals, rows, cols

    # 'sample_data' is a sparse matrix with 4 columns: ('action_ind', 'atom_map1', 'atom_map2', 'n_nodes')
    sample_data = [], [], []  # vals, rows, cols
    action_tuples = []

    metadata = {
        'reaction_ind': [],
        'is_train': [],
        'n_samples': [],
        'start_ind': [],
        'final_smi': [],
    }
    if 'class' in data_x:
        metadata['class'] = []

    if forward:
        target_x = data_x['product']
        source_x = data_x['substrates']
    else:
        target_x = data_x['substrates']
        source_x = data_x['product']

    n_unparsed = 0

    for reaction_i, (reaction_ind, train, target_smi, source_smi) in \
            feat_loop(enumerate(zip(data_inds, is_train, target_x, source_x)),
                      desc='Thread {}: converting reactions to training samples...'.format(thread_num),
                      total=n_reactions):
        if not isinstance(target_smi, str) or len(target_smi) == 0:
            target_smi = source_smi

        try:
            target_mol = Chem.MolFromSmiles(target_smi)
            source_mol = Chem.MolFromSmiles(source_smi)

            if target_mol is not None and source_mol is not None:
                target_mol, source_mol = fix_incomplete_mappings(target_mol, source_mol)
                target_mol, source_mol = reac_to_canonical(target_mol, source_mol)
                source_mol = fix_explicit_hs(source_mol)
                target_mol = fix_explicit_hs(target_mol)

                source_mol = renumber_atoms_for_mapping(source_mol)
                target_mol = renumber_atoms_for_mapping(target_mol)
        except Exception:
            target_mol, source_mol = None, None

        if target_mol is None or source_mol is None:
            n_unparsed += 1
            continue

        try:
            training_samples, final_smi = gen_training_samples(target_mol, source_mol, n_max_steps,
                                                               action_vocab=action_vocab,
                                                               forward=forward, action_order=action_order)
        except Exception:
            n_unparsed += 1
            continue

        if len(training_samples) == 0:
            n_unparsed += 1
            continue

        start_ind = reaction_ind * n_max_steps

        metadata['reaction_ind'].append(reaction_ind)
        metadata['is_train'].append(train)
        metadata['final_smi'].append(str(final_smi))
        metadata['start_ind'].append(start_ind)
        metadata['n_samples'].append(len(training_samples))

        if 'class' in data_x:
            metadata['class'].append(data_x['class'][reaction_i])

        for sample_ind, sample in enumerate(training_samples):
            ind = start_ind + sample_ind
            action_tuples.append((ind, sample['action_tuple']))

            sample_data[0].append(sample['atom_map1'])
            sample_data[1].append(ind)
            sample_data[2].append(1)

            sample_data[0].append(sample['atom_map2'])
            sample_data[1].append(ind)
            sample_data[2].append(2)

            nodes = sample['nodes']
            sample_data[0].append(len(nodes))
            sample_data[1].append(ind)
            sample_data[2].append(3)

            sample_data[0].append(int(sample['is_hard']))
            sample_data[1].append(ind)
            sample_data[2].append(4)

            if rt_given:
                reaction_type = data_x['reaction_type'][reaction_i]
                sample_data[0].append(reaction_type)
                sample_data[1].append(ind)
                sample_data[2].append(5)

            for j, node in enumerate(nodes):
                nodes_vals[0].append(node)
                nodes_vals[1].append(ind)
                nodes_vals[2].append(j)

            for val, row, col in zip(sample['adj'][0], sample['adj'][1], sample['adj'][2]):
                adj_vals[0].append(val)
                adj_vals[1].append(ind)
                adj_vals[2].append(row * max_n_nodes + col)

        if reaction_ind > 0 and reaction_ind % k == 0:
            nodes_mat += sparse.csr_matrix((nodes_vals[0], (nodes_vals[1], nodes_vals[2])),
                                           shape=(samples_len, max_n_nodes))
            adj_mat += sparse.csr_matrix((adj_vals[0], (adj_vals[1], adj_vals[2])),
                                         shape=(samples_len, max_n_nodes ** 2))
            adj_vals = [], [], []
            nodes_vals = [], [], []

    if len(nodes_vals[0]) > 0:
        nodes_mat += sparse.csr_matrix((nodes_vals[0], (nodes_vals[1], nodes_vals[2])),
                                       shape=(samples_len, max_n_nodes))
        adj_mat += sparse.csr_matrix((adj_vals[0], (adj_vals[1], adj_vals[2])),
                                     shape=(samples_len, max_n_nodes ** 2))

    n_sample_data = 6 if rt_given else 5
    sample_data = sparse.csr_matrix((sample_data[0], (sample_data[1], sample_data[2])),
                                    shape=(samples_len, n_sample_data))

    if not os.path.exists(chunk_save_path):
        os.makedirs(chunk_save_path)

    actions_save_path = os.path.join(chunk_save_path, 'actions.txt')
    with open(actions_save_path, 'w') as fp:
        for action in action_tuples:
            fp.write(str(action))
            fp.write('\n')

    meta_save_path = os.path.join(chunk_save_path, 'metadata.csv')
    pd.DataFrame(metadata).to_csv(meta_save_path)

    sample_data_path = os.path.join(chunk_save_path, 'sample_data.npz')
    sparse.save_npz(sample_data_path, sample_data)

    nodes_mat_path = os.path.join(chunk_save_path, 'nodes_mat.npz')
    sparse.save_npz(nodes_mat_path, nodes_mat)

    adj_mat_path = os.path.join(chunk_save_path, 'adj_mat.npz')
    sparse.save_npz(adj_mat_path, adj_mat)

    return n_unparsed
