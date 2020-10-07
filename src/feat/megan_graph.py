# -*- coding: utf-8 -*-
"""
Generates featurized training/validation samples for training MEGAN reaction generation model.
Each sample represents a single edit of the target molecule (for training for retrosynthesis).
Training can be stateful or stateless.
"""
import json
import logging
import os
import shutil
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from src.feat import ReactionFeaturizer

from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS

from src.feat.graph_features import ATOM_PROPS, BOND_PROPS, ATOM_PROP2OH, BOND_PROP2OH
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

from src.datasets import Dataset
from src.feat.featurize import featurize_parallel
from src.feat.find_properties import find_properties_parallel
from src.feat.mol_graph import unravel_atom_features, unravel_bond_features
from src.split import DatasetSplit
from src.utils import to_torch_tensor, lists_to_tuple

logger = logging.getLogger(__name__)

# there is a warning about hydrogen atoms that do not have neighbors that could not be deleted (that is OK)
RDLogger.DisableLog('rdApp.*')


def get_adj_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'adj.npz')


def get_nodes_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'nodes.npz')


def get_sample_data_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'sample_data.npz')


def get_metadata_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'metadata.csv')


def get_actions_vocab_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'all_actions.json')


def get_prop2oh_vocab_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'prop2oh.json')


class MeganTrainingSamplesFeaturizer(ReactionFeaturizer):
    """
    Converts a mapped reaction SMILES to series of training/validation samples.
    """

    def __init__(self, split: DatasetSplit, n_jobs: int = 1,
                 key: str = 'megan', max_n_steps: int = 32,
                 forward: bool = False, action_order: str = 'dfs'):
        """
        :param n_jobs: number of threads
        :param key: key of the featurizer
        :param max_n_steps: maximum number of generation steps
        :param split: split to use to filter out 'test' samples
        :param forward: generate samples for learning forward synthesis instead of backward
        :param action_order: type of action ordering ('dfs'/'bfs'/'dfs_random'/'bfs_random','random')
        """
        super(MeganTrainingSamplesFeaturizer, self).__init__()
        assert max_n_steps > 0
        assert n_jobs != 0

        self.n_jobs = n_jobs
        for_str = '_for' if forward else ''
        self.key = f'{key}{for_str}_{split.key}_{max_n_steps}_{action_order}'

        self.max_n_steps = max_n_steps
        self.split = split
        self._vocabulary = None
        self.forward = forward
        self.action_order = action_order

    def dir(self, feat_dir: str) -> str:
        return os.path.join(feat_dir, self.key)

    def has_finished(self, feat_dir: dir) -> bool:
        this_feat_dir = self.dir(feat_dir)
        return all(os.path.exists(get_path(this_feat_dir)) for get_path in
                   (get_adj_path, get_nodes_path, get_metadata_path, get_sample_data_path))

    def get_actions_vocabulary(self, feat_dir: str) -> dict:
        if self._vocabulary is None:
            with open(get_actions_vocab_path(self.dir(feat_dir)), 'r') as fp:
                action_tuples = json.load(fp)

            prop2oh_path = get_prop2oh_vocab_path(self.dir(feat_dir))
            if not os.path.exists(prop2oh_path):
                # use default OH feature values (for backward compatibility)
                props = {'atom': ATOM_PROPS, 'bond': BOND_PROPS}
                prop2oh = {'atom': ATOM_PROP2OH, 'bond': BOND_PROP2OH}
            else:
                with open(prop2oh_path, 'r') as fp:
                    props_json = json.load(fp)
                props = {'atom': props_json['atom'], 'bond': props_json['bond']}
                prop2oh = {'atom': {}, 'bond': {}}
                # convert string keys to integers if possible
                for type_key in prop2oh.keys():
                    oh_dict = props_json[f'{type_key}_2oh']
                    for key, values in oh_dict.items():
                        converted_values = {}
                        for prop_val, val_oh in values.items():
                            try:
                                prop_val = int(prop_val)
                            except ValueError:
                                pass
                            converted_values[prop_val] = val_oh
                        prop2oh[type_key][key] = converted_values

            action_tuples = [lists_to_tuple(a) for a in action_tuples]

            # add special shortcut values for quick usage in evaluation
            action2ind = dict((k, i) for i, k in enumerate(action_tuples))
            vocab = {
                'props': props,
                'prop2oh': prop2oh,
                'action_tuples': action_tuples,
                'action2ind': action2ind,
                'atom_actions': [action for action in action_tuples if action[0] == 'change_atom'
                                 or action[0] == 'add_ring' or action[0] == 'add_atom' or action[0] == 'stop'],
                'bond_actions': [action for action in action_tuples if action[0] == 'change_bond'],
            }
            vocab['n_atom_actions'] = len(vocab['atom_actions'])
            vocab['n_bond_actions'] = len(vocab['bond_actions'])
            vocab['n_target_actions'] = max(vocab['n_atom_actions'], vocab['n_bond_actions'])
            vocab['atom_action_num'] = dict((action2ind[action], i) for i, action in enumerate(vocab['atom_actions']))
            vocab['bond_action_num'] = dict((action2ind[action], i) for i, action in enumerate(vocab['bond_actions']))
            vocab['stop_action_num'] = vocab['atom_action_num'][action2ind[('stop',)]]

            atom_feature_keys = [k for k in ORDERED_ATOM_OH_KEYS if k in vocab['prop2oh']['atom']]
            bond_feature_keys = [k for k in ORDERED_BOND_OH_KEYS if k in vocab['prop2oh']['bond']]
            vocab['atom_feature_keys'] = atom_feature_keys
            vocab['bond_feature_keys'] = bond_feature_keys

            vocab['atom_feat_ind'] = dict((k, i) for i, k in enumerate(atom_feature_keys))
            vocab['bond_feat_ind'] = dict((k, i) for i, k in enumerate(bond_feature_keys))

            self._vocabulary = vocab

        return self._vocabulary

    def featurize_dataset(self, dataset: Dataset):
        logger.info(f"Loading dataset {dataset.key} and {self.split.key} split")
        data = dataset.load_x()
        for required_field in ['product', 'substrates']:
            if required_field not in data:
                raise NotImplementedError(f"Need to have field '{required_field} in the dataset")

        split = self.split.load(dataset.dir)
        feat_dir = self.dir(dataset.feat_dir)

        metadata = dataset.load_metadata()
        reaction_type_given = False
        if 'reaction_type_id' in metadata:
            rtypes = metadata['reaction_type_id'].values
            ntypes = len(np.unique(rtypes))
            logger.info(f'Found {ntypes} unique reaction types in the dataset')
            reaction_type_given = True
            data['reaction_type'] = rtypes

        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        if 'max_n_nodes' in dataset.meta_info:
            max_n_nodes = dataset.meta_info['max_n_nodes']
        else:
            max_n_nodes = 1024
        logger.info("Max. number of nodes: {}".format(max_n_nodes))

        # we do not featurize test set for training
        all_inds = np.argwhere(split['test'] == 0).flatten()

        # shuffle indices for featurization in multiple threads
        np.random.shuffle(all_inds)

        data_len = len(data)
        samples_len = data_len * self.max_n_steps

        chunk_size = int(len(all_inds) / self.n_jobs)
        chunk_ends = [chunk_size * i for i in range(self.n_jobs + 1)]
        chunk_ends[-1] = len(all_inds)
        chunk_inds = [all_inds[chunk_ends[i]: chunk_ends[i + 1]] for i in range(len(chunk_ends) - 1)]

        logger.info(f'Finding all possible values of atom and bond properties '
                    f'on {len(all_inds)} reactions using {self.n_jobs} chunks')
        parallel_args = []
        for i, ch_inds in enumerate(chunk_inds):
            new_x = dict((k, x.values[ch_inds]) for k, x in data.items())
            parallel_args.append((i, new_x, tqdm))

        prop_dict = {'atom': {}, 'bond': {}}
        if self.n_jobs == 1:
            chunk_results = [find_properties_parallel(parallel_args[0])]
        else:
            pool = Pool(self.n_jobs)
            chunk_results = pool.imap(find_properties_parallel, parallel_args)

        for chunk_prop_dict in chunk_results:
            for type_key in prop_dict.keys():
                for key, values in chunk_prop_dict[type_key].items():
                    if key not in prop_dict[type_key]:
                        prop_dict[type_key][key] = set()
                    prop_dict[type_key][key].update(values)

        # add some 'special' atom/bond feature values
        prop_dict['atom']['is_supernode'].update([0, 1])
        prop_dict['atom']['is_edited'].update([0, 1])
        prop_dict['atom']['is_reactant'].update([0, 1])
        prop_dict['bond']['bond_type'].update(['supernode', 'self'])
        prop_dict['bond']['is_edited'].update([0, 1])

        atom_feat_counts = ', '.join(['{:s}: {:d}'.format(key, len(values))
                                      for key, values in prop_dict['atom'].items()])
        logger.info(f'Found atom features: {atom_feat_counts}')

        bond_feat_counts = ', '.join(['{:s}: {:d}'.format(key, len(values))
                                      for key, values in prop_dict['bond'].items()])
        logger.info(f'Found bond features: {bond_feat_counts}')

        # make a dictionary for conversion of atom/bond features to OH numbers
        prop2oh = {'atom': {}, 'bond': {}}
        props = {'atom': {}, 'bond': {}}
        for type_key, prop_values in prop_dict.items():
            for prop_key, values in prop_values.items():
                sorted_vals = list(sorted(values, key=lambda x: x if isinstance(x, int) else 0))
                props[type_key][prop_key] = sorted_vals
                oh = dict((k, i + 1) for i, k in enumerate(sorted_vals))
                prop2oh[type_key][prop_key] = oh

        # save 'prop2oh' dictionary
        with open(get_prop2oh_vocab_path(feat_dir), 'w') as fp:
            json.dump({'atom': props['atom'], 'bond': props['bond'],
                       'atom_2oh': prop2oh['atom'], 'bond_2oh': prop2oh['bond']}, fp, indent=2)

        atom_feature_keys = [k for k in ORDERED_ATOM_OH_KEYS if k in prop2oh['atom']]
        bond_feature_keys = [k for k in ORDERED_BOND_OH_KEYS if k in prop2oh['bond']]
        action_vocab = {
            'prop2oh': prop2oh,
            'atom_feature_keys': atom_feature_keys,
            'bond_feature_keys': bond_feature_keys,
            'atom_feat_ind': dict((k, i) for i, k in enumerate(atom_feature_keys)),
            'bond_feat_ind': dict((k, i) for i, k in enumerate(bond_feature_keys))
        }

        parallel_args = []
        chunk_save_paths = []
        for i, ch_inds in enumerate(chunk_inds):
            new_x = dict((k, x.values[ch_inds]) for k, x in data.items())
            is_train = split['train'][ch_inds].values
            chunk_save_path = os.path.join(feat_dir, f'chunk_result_{i}')
            chunk_save_paths.append(chunk_save_path)
            parallel_args.append((i, samples_len, ch_inds, new_x, max_n_nodes, tqdm,
                                  self.max_n_steps, is_train, reaction_type_given,
                                  self.forward, self.action_order,
                                  action_vocab, chunk_save_path))

        logger.info(f'Featurizing {len(all_inds)} reactions with {self.n_jobs} threads')
        logger.info(f"Number of generated paths (train+valid): {data_len}")
        logger.info(f"Upper bound for number of generated samples: {samples_len} ({data_len} * {self.max_n_steps})")

        if self.n_jobs == 1:
            chunk_results = [featurize_parallel(parallel_args[0])]
        else:
            # leave one job for merging results
            pool = Pool(max(self.n_jobs - 1, 1))
            chunk_results = pool.imap(featurize_parallel, parallel_args)

        logger.info(f"Merging featurized data from {self.n_jobs} chunks")

        nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes))
        adj_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes ** 2))

        n_sample_data = 6 if reaction_type_given else 5
        sample_data_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, n_sample_data))
        meta = []

        # vocabulary of actions
        actions_vocab = []
        action2ind = {}
        action_inds = []
        action_tuples = []
        sample_inds = []

        for ch_inds, result_code, chunk_save_path in tqdm(zip(chunk_inds, chunk_results, chunk_save_paths),
                                                          desc='merging reactions from chunks', total=self.n_jobs):
            sample_data_path = os.path.join(chunk_save_path, 'sample_data.npz')
            sample_data_mat += sparse.load_npz(sample_data_path)

            nodes_mat_path = os.path.join(chunk_save_path, 'nodes_mat.npz')
            nodes_mat += sparse.load_npz(nodes_mat_path)

            adj_mat_path = os.path.join(chunk_save_path, 'adj_mat.npz')
            adj_mat += sparse.load_npz(adj_mat_path)

            meta_save_path = os.path.join(chunk_save_path, 'metadata.csv')
            chunk_meta = pd.read_csv(meta_save_path)
            meta.append(chunk_meta)

            actions_save_path = os.path.join(chunk_save_path, 'actions.txt')
            chunk_action_tuples = []
            for line in open(actions_save_path, 'r'):
                action = eval(line.strip())
                chunk_action_tuples.append(action)

            for sample_ind, action in chunk_action_tuples:
                if action in action2ind:
                    action_inds.append(action2ind[action])
                else:
                    action_ind = len(actions_vocab)
                    action2ind[action] = action_ind
                    actions_vocab.append(action)
                    action_tuples.append(action)
                    action_inds.append(action_ind)
                sample_inds.append(sample_ind)

            # remove temporary chunk files
            shutil.rmtree(chunk_save_path)
            logger.info(f"Merged chunk {len(meta)} (unparsed samples: {result_code}/{len(ch_inds)})")

        logger.info("Concatenating metadata")
        meta = pd.concat(meta)

        logger.info("Saving found actions")
        sample_data_mat[sample_inds, 0] = action_inds
        with open(get_actions_vocab_path(feat_dir), 'w') as fp:
            json.dump(action_tuples, fp)
        logger.info(f"Found {len(action_tuples)} reaction actions")

        n_samples = meta['n_samples']
        logger.info(f"Number of steps: max: {np.max(n_samples)}, avg: {np.mean(n_samples)}")

        logger.info("Saving featurized data")
        meta.to_csv(get_metadata_path(feat_dir))
        sparse.save_npz(get_sample_data_path(feat_dir), sample_data_mat)
        sparse.save_npz(get_nodes_path(feat_dir), nodes_mat)
        sparse.save_npz(get_adj_path(feat_dir), adj_mat)

        n_saved_reacs = len(np.unique(meta['reaction_ind']))

        logger.info(f"Saved {n_saved_reacs}/{len(all_inds)} reactions ({n_saved_reacs / len(all_inds) * 100}%)")
        logger.info(f"Saved {len(meta)} paths (avg. {len(meta) / n_saved_reacs} paths per reaction)")

        logger.info("Saving featurization metadata")
        meta_info = {
            'description': 'Graph representation of molecules with discrete node and edge features for MEGAN',
            'features': ['atom', 'bond'],
            'features_type': ['atom', 'bond'],
            'max_n_nodes': max_n_nodes,
            'format': 'sparse'
        }
        meta_path = self.meta_info_path(dataset.feat_dir)
        with open(meta_path, 'w') as fp:
            json.dump(meta_info, fp, indent=2)

    def featurize_batch(self, metadata_dir: str, batch: dict) -> dict:
        raise NotImplementedError("TODO")

    def load(self, feat_dir: str) -> dict:
        this_feat_dir = self.dir(feat_dir)
        result = {
            'reaction_metadata': pd.read_csv(get_metadata_path(this_feat_dir)),
            'atom': sparse.load_npz(get_nodes_path(this_feat_dir)),
            'bond': sparse.load_npz(get_adj_path(this_feat_dir)),
            'sample_data': sparse.load_npz(get_sample_data_path(this_feat_dir))
        }
        return result

    # noinspection PyMethodOverriding
    def to_tensor_batch(self, data: dict, actions_vocab: dict) -> dict:
        batch_max_nodes = data['max_n_nodes']  # number of nodes in each graph in batch
        props = actions_vocab['props']

        nodes = data['atom'][:, :batch_max_nodes]
        if hasattr(nodes, 'toarray'):
            nodes = nodes.toarray()
        nodes = nodes.astype(int)

        edges = data['bond']
        if hasattr(edges, 'toarray'):
            edges = edges.toarray()
        max_n = int(np.sqrt(edges.shape[-1]))
        edges = edges.reshape(edges.shape[0], max_n, max_n)
        edges = edges[:, :batch_max_nodes, :batch_max_nodes].astype(int)

        # unravel discrete features
        node_oh_dim = [len(props['atom'][feat_key]) + 1 for feat_key in ORDERED_ATOM_OH_KEYS]
        unraveled_nodes = unravel_atom_features(nodes, node_oh_dim=node_oh_dim)
        unraveled_nodes = unraveled_nodes.transpose(1, 2, 0)
        data['atom'] = to_torch_tensor(unraveled_nodes, long=True)

        edge_oh_dim = [len(props['bond'][feat_key]) + 1 for feat_key in ORDERED_BOND_OH_KEYS]
        unraveled_edges = unravel_bond_features(edges, edge_oh_dim=edge_oh_dim)
        unraveled_edges = unraveled_edges.transpose(1, 2, 3, 0)
        data['bond'] = to_torch_tensor(unraveled_edges, long=True)

        if 'reaction_type' in data:
            data['reaction_type'] = to_torch_tensor(data['reaction_type'], long=True)

        return data
