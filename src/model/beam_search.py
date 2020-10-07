"""
Scripts for performing beam search on MEGAN
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from src.feat import ORDERED_BOND_OH_KEYS, ORDERED_ATOM_OH_KEYS
from src.feat.mol_graph import get_graph
from src.feat.reaction_actions import ReactionAction, StopAction, AtomEditAction, AddAtomAction, AddRingAction, \
    BondEditAction
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.model.megan import Megan
from src.model.megan_utils import generate_eval_batch, mols_from_graph, RdkitCache

logger = logging.getLogger(__name__)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_top_k_paths(paths: List[dict], beam_size: int, n_mols: int, sort: bool = True):
    filtered_paths = []
    for i in range(n_mols):
        mol_paths = [p for p in paths if p['mol_ind'] == i]
        if sort:
            path_argsort = np.argsort([-p['prob'] for p in mol_paths])
            mol_paths = [mol_paths[j] for j in path_argsort[:beam_size]]
        else:
            mol_paths = mol_paths[:beam_size]
        filtered_paths += mol_paths
    return filtered_paths


def paths_are_probably_same(path1: dict, path2: dict, eps: float = 1e-6) -> bool:
    if abs(path1['prob'] - path2['prob']) < eps:
        return True
    if path1['action_set'] == path2['action_set']:
        return True
    return False


def filter_duplicate_paths(paths: List[dict], n_mols: int):
    filter_paths = np.full(len(paths), fill_value=True, dtype=bool)

    for path in paths:
        path['action_set'] = set(str(a) for a, p in path['actions'])

    for mol_i in range(n_mols):
        mol_paths_i = [i for i in range(len(paths)) if paths[i]['mol_ind'] == mol_i]
        mol_paths = [paths[i] for i in mol_paths_i]
        
        is_unique = np.full(len(mol_paths), fill_value=True, dtype=bool)
        for path_i in range(len(mol_paths)):
            for prev_path in mol_paths[:path_i]:
                if paths_are_probably_same(prev_path, mol_paths[path_i]):
                    is_unique[path_i] = False
                    prev_path['prob'] += mol_paths[path_i]['prob']
                    mol_paths[path_i]['prob'] = prev_path['prob']
                    break
        filter_paths[mol_paths_i] = is_unique
    paths = [paths[i] for i, is_unique in enumerate(filter_paths) if is_unique]
    return paths


def get_batch(paths: List[dict], base_action_masks: dict,
              reaction_types: Optional[np.ndarray] = None) -> Tuple[dict, List[torch.Tensor]]:
    batch = generate_eval_batch([p['mol_graph'] for p in paths],
                                base_action_masks=base_action_masks, reaction_types=reaction_types)
    batch['max_map_num'] = [p['max_map_num'] for p in paths]

    models_step_states = []
    for model_i in range(len(paths[0]['state'])):
        if paths[0]['state'][model_i] is None:
            step_state = None
        else:
            stacked_shape = [len(paths),
                             max(p['state'][model_i].shape[0] for p in paths),
                             paths[0]['state'][model_i].shape[1]]
            stacked_tensors = torch.zeros(stacked_shape, device=device)
            for i, p in enumerate(paths):
                p_val = p['state'][model_i]
                stacked_tensors[i, :p_val.shape[0]] = p_val
            step_state = stacked_tensors
        models_step_states.append(step_state)

    return batch, models_step_states


def tuple2action(action: Tuple, atom1: int, atom2: int, max_map_num: int, action_vocab: dict) -> ReactionAction:
    action_type = action[0]

    if action_type == 'stop':
        return StopAction(action_vocab=action_vocab)
    elif action_type == 'change_atom':
        return AtomEditAction(atom1, *action[1], action_vocab=action_vocab)
    elif action_type == 'add_atom':
        atom2 = max_map_num + 1
        return AddAtomAction(atom1, atom2, *action[1][0], *action[1][1], action_vocab=action_vocab)
    elif action_type == 'add_ring':
        new_atoms_map_nums = [atom1] + [max_map_num + i + 1 for i in range(5)]
        return AddRingAction(atom1, new_atoms_map_nums, action[1], action_vocab=action_vocab)
    elif action_type == 'change_bond':
        return BondEditAction(atom1, atom2, *action[1], action_vocab=action_vocab)
    else:
        raise ValueError(f'Unknown action type: {action_type}')


def get_topk(output, mask, beam_size):
    nonzero_where = torch.nonzero(mask, as_tuple=False)[:, 0]
    n_nonzeros = nonzero_where.shape[0]

    if n_nonzeros == 0:
        return torch.tensor((), dtype=torch.float, device=device), \
               torch.tensor((), dtype=torch.long, device=device)
    else:
        beam_size = min(beam_size, output[mask].shape[0])
        action_probs, action_numbers = torch.topk(output[mask], beam_size, sorted=False)

    action_numbers = nonzero_where[action_numbers]
    return action_probs, action_numbers


def get_best_actions(output: torch.Tensor, n_nodes: int, beam_size: int, action_vocab: dict,
                     min_prob: float = 0.0, min_stop_prob: float = 0.0) \
        -> List[List[Tuple[float, Tuple, int]]]:
    all_probs, all_nums, n_actions = [], [], []
    nonzero_mask = output > min_prob
    action_shape = (1, n_nodes + 1, n_nodes, action_vocab['n_target_actions'])

    if min_stop_prob > 0:
        stop_ind = (0, n_nodes, 0, action_vocab['stop_action_num'])
        stop_ind = np.ravel_multi_index(stop_ind, action_shape)
        nonzero_mask[:, stop_ind] *= output[:, stop_ind] > min_stop_prob

    for sample_i in range(output.shape[0]):
        action_probs, action_numbers = get_topk(output[sample_i], nonzero_mask[sample_i], beam_size)
        all_probs.append(action_probs)
        all_nums.append(action_numbers)

    n_actions = [len(a) for a in all_probs]
    action_probs = torch.cat(all_probs).cpu().numpy()
    action_numbers = torch.cat(all_nums).cpu().numpy()
    unraveled_ind = np.stack(np.unravel_index(action_numbers, action_shape), axis=0).T

    if len(n_actions) == 0:
        return [[]]

    result = []
    begin_i, end_i = 0, 0
    for sample_i in range(output.shape[0]):
        begin_i = end_i
        end_i = end_i + n_actions[sample_i]
        ind_range = np.arange(begin_i, end_i)
        result.append([(action_probs[ind], tuple(unraveled_ind[ind]), n_nodes) for ind in ind_range])
    return result


def get_action_object(action_inds: Tuple, n_nodes: int, max_map_num: int, action_vocab: dict) -> ReactionAction:
    action_number = action_inds[-1]

    if action_inds[1] == n_nodes:
        action = action_vocab['atom_actions'][action_number]
        node_ind1, node_ind2 = int(action_inds[2]), -1
    else:
        action = action_vocab['bond_actions'][action_number]
        node_ind1, node_ind2 = int(action_inds[1]), int(action_inds[2])

    return tuple2action(action, node_ind1, node_ind2, max_map_num, action_vocab)


def beam_search(models: List[Megan], mols: List[Mol], base_action_masks: dict, action_vocab: dict,
                rdkit_cache: RdkitCache, max_steps: int = 16, beam_size: int = 1,
                batch_size: int = 32, max_atoms: int = 200, min_prob: float = 0.0, min_stop_prob: float = 0.0,
                filter_duplicates: bool = False, filter_incorrect: bool = True,
                reaction_types: Optional[np.ndarray] = None, softmax_base: float = 1.0,
                export_samples: bool = False, only_last: bool = False) -> List[List[dict]]:
    paths = []

    bond_dirs = []
    for i, input_mol in enumerate(mols):
        mol_bond_dirs = {}
        if input_mol is not None and input_mol.GetNumAtoms() > 0:
            mol_graph = get_graph(input_mol, ravel=False, to_array=True,
                                  atom_prop2oh=action_vocab['prop2oh']['atom'],
                                  bond_prop2oh=action_vocab['prop2oh']['bond'])

            start_path = {
                'n_steps': 0,
                'prob': 1.0,
                'mol_graph': mol_graph,
                'max_map_num': max(a.GetAtomMapNum() for a in input_mol.GetAtoms()),
                'state': [None for _ in range(len(models))],
                'actions': [],
                'finished': False,
                'changed_atoms': set(),
                'mol_ind': i
            }
            if export_samples:
                start_path['mol_graphs'] = []

            paths.append(start_path)

            # save information about Bond Direction. This is needed to replicate isomer information from input mol.
            # this is cached here to speedup later "mol_to_graph" operation
            for bond in input_mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_dir = bond.GetBondDir()
                if a1 > a2:
                    a1, a2 = a2, a1
                mol_bond_dirs[(a1, a2)] = bond_dir

        bond_dirs.append(mol_bond_dirs)

    edge_oh_dim = [len(action_vocab['prop2oh']['bond'][feat_key]) + 1 for feat_key in ORDERED_BOND_OH_KEYS if feat_key in action_vocab['prop2oh']['bond']]
    node_oh_dim = [len(action_vocab['prop2oh']['atom'][feat_key]) + 1 for feat_key in ORDERED_ATOM_OH_KEYS if feat_key in action_vocab['prop2oh']['atom']]

    def ravel_graph(adj, nodes):
        ravel_nodes = np.ravel_multi_index(nodes.T, node_oh_dim)
        ravel_adj = np.ravel_multi_index(adj.reshape(-1, adj.shape[-1]).T, edge_oh_dim)
        ravel_adj = ravel_adj.reshape((adj.shape[0], adj.shape[1]))

        return ravel_adj, ravel_nodes

    def process_paths_batch(path_batch, batch_beam_size=beam_size) -> List[dict]:
        if reaction_types is not None:
            batch_reaction_types = np.asarray([reaction_types[p['mol_ind']] for p in path_batch])
        else:
            batch_reaction_types = None
        step_batch, step_state = get_batch(path_batch, base_action_masks, batch_reaction_types)
        step_batch['base'] = softmax_base
        new_batch_paths = []

        try:
            # ensemble predictions from models
            step_results = {'state': [], 'output': [], 'output_mask': []}

            for model_i, model in enumerate(models):
                state_dict = None if step_state[model_i] is None else {'state': step_state[model_i]}
                model_batch = dict((k, v) for k, v in step_batch.items())
                model_step_results = model.forward_step(model_batch, state_dict=state_dict)
                for key, val in model_step_results.items():
                    if key not in step_results:
                        step_results[key] = []
                    step_results[key].append(val)

            step_results['output_mask'] = torch.max(torch.stack(step_results['output_mask']), dim=0)[0]
            step_results['output'] = torch.mean(torch.stack(step_results['output']), dim=0)
            if 'class_output' in step_results:
                step_results['class_output'] = torch.mean(torch.stack(step_results['class_output']), dim=0)
                step_results['class_output'] = step_results['class_output'].cpu().detach().numpy()

        except Exception as e:
            logger.warning(f"Exception while 'model.forward_step': {str(e)}")
            return new_batch_paths

        batch_actions = get_best_actions(step_results['output'], step_batch['adj'].shape[1],
                                         batch_beam_size, action_vocab, min_prob, min_stop_prob)

        for path_i, path in enumerate(path_batch):
            max_map_num = step_batch['max_map_num'][path_i]
            if max_map_num > max_atoms:
                continue
            actions = batch_actions[path_i]
            n_steps = path['n_steps'] + 1
            state = [step_results['state'][model_i][path_i] for model_i in range(len(models))]

            for action_prob, action_inds, n_nodes in actions:
                last_action = action_prob, action_inds, n_nodes
                new_prob = path['prob'] * action_prob

                if action_inds[1] == n_nodes and action_inds[-1] == action_vocab['stop_action_num']:
                    final_path = {
                        'finished': True,
                        'prob': new_prob,
                        'n_steps': n_steps,
                        'actions': path['actions'] + [(StopAction(action_vocab=action_vocab),
                                                       action_prob)],
                        'changed_atoms': path['changed_atoms'],
                        'mol_graph': path['mol_graph'],
                        'state': state ,
                        'mol_ind': path['mol_ind']
                    }
                    if 'class_output' in step_results:
                        final_path['class_output'] = step_results['class_output'][path_i]
                    if export_samples:
                        final_path['mol_graphs'] = path['mol_graphs'] + [ravel_graph(*path['mol_graph'])]
                    new_batch_paths.append(final_path)
                else:
                    new_path = {
                        'finished': False,
                        'n_steps': n_steps,
                        'prob': new_prob,
                        'mol_graph': path['mol_graph'],
                        'max_map_num': max_map_num,
                        'state': state ,
                        'actions': path['actions'],
                        'last_action': last_action,
                        'changed_atoms': path['changed_atoms'],
                        'mol_ind': path['mol_ind']
                    }
                    if export_samples:
                        new_path['mol_graphs'] = path['mol_graphs'] + [ravel_graph(*path['mol_graph'])]
                    new_batch_paths.append(new_path)

        return new_batch_paths

    for step_i in range(max_steps):
        # apply last actions for all paths
        for path in paths:
            if path['finished'] or 'last_action' not in path:  # first step of generation or finished
                continue

            action_prob, action_inds, n_nodes = path['last_action']
            max_map_num = path['max_map_num']

            action = get_action_object(action_inds, n_nodes, max_map_num, action_vocab=action_vocab)

            changed_atoms = path['changed_atoms']
            changed_atoms = changed_atoms.copy()
            if action.atom_map1 > 0:
                changed_atoms.add(action.atom_map1)
            if action.atom_map2 > 0:
                changed_atoms.add(action.atom_map2)
            if isinstance(action, AddRingAction):
                for map_num in action.new_atoms_map_nums:
                    changed_atoms.add(map_num)

            if len(changed_atoms) > 0:
                path['max_map_num'] = max(max(changed_atoms), max_map_num)
            path['changed_atoms'] = changed_atoms
            path['actions'] = path['actions'] + [(action, action_prob)]

            try:
                path['mol_graph'] = action.graph_apply(*path['mol_graph'])
            except Exception as e:
                logger.warning(f'Exception during applying action: {str(e)}')
                continue

        # find paths with duplicate graphs
        if step_i > 0 and filter_duplicates:
            paths = filter_duplicate_paths(paths, len(mols))

        analyzed_paths = [path for path in paths if not path['finished']]
        if len(analyzed_paths) == 0:
            break

        paths = [path for path in paths if path['finished']]

        n_batches = int(np.ceil(len(analyzed_paths) / batch_size))
        path_batches = np.array_split(analyzed_paths, n_batches)

        for p_batch in path_batches:
            new_paths = process_paths_batch(p_batch, beam_size)
            paths += new_paths

        # sort paths by probabilities and limit number of paths
        paths = get_top_k_paths(paths, beam_size, len(mols))

        if all(p['finished'] for p in paths):
            break

    finished_paths = [[] for _ in range(len(mols))]
    if only_last:
        paths_iter = reversed(paths)
    else:
        paths_iter = paths

    for path in paths_iter:
        ind = path['mol_ind']
        if only_last and len(finished_paths[ind]) > 0:
            continue
        if path['finished']:
            try:
                adj, nodes = path['mol_graph']
                mol_ind = path['mol_ind']
                output_mols = mols_from_graph(rdkit_cache, mols[mol_ind], bond_dirs[mol_ind], adj, nodes,
                                              changed_atoms=path['changed_atoms'], only_edited=False)
                final_smi = '.'.join([Chem.MolToSmiles(mol) for mol in output_mols if mol is not None])

                for mol in output_mols:
                    if mol:
                        for a in mol.GetAtoms():
                            a.ClearProp("molAtomMapNumber")
                final_smi_unm = '.'.join([Chem.MolToSmiles(mol) for mol in output_mols if mol is not None])

            except Exception as e:
                # incorrect final mol
                logger.debug(f'Exception while final mol to smiles: {str(e)}')
                final_smi, final_smi_unm = '', ''

            if not filter_incorrect or final_smi_unm:
                result_path = {
                    'final_smi': final_smi,
                    'final_smi_unmapped': final_smi_unm,
                    'prob': path['prob'],
                    'actions': path['actions'],
                    'n_steps': path['n_steps'],
                    'changed_atoms': path['changed_atoms'],
                    'mol_graph': path['mol_graph']
                }
                if 'class_output' in path:
                    result_path['class_output'] = path['class_output']
                if export_samples:
                    result_path['mol_graphs'] = path['mol_graphs']

                # noinspection PyTypeChecker
                finished_paths[ind].append(result_path)

    return finished_paths
