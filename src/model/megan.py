"""
MEGAN model for reaction generation
"""
from typing import Tuple, Optional, List

import gin
import torch
from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS
from torch import nn
from torch.autograd import Variable

from src.model.megan_modules.decoder import MeganDecoder
from src.model.megan_modules.encoder import MeganEncoder

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def to_one_hot(x, dims: int):
    one_hot = torch.FloatTensor(*x.shape, dims).zero_().to(device)
    x = torch.unsqueeze(x, -1)
    target = one_hot.scatter_(x.dim() - 1, x.data, 1)

    target = Variable(target)
    return target


default_atom_features = 'is_supernode', 'atomic_num', 'formal_charge', 'chiral_tag', \
                        'num_explicit_hs', 'is_aromatic', 'is_edited'
default_bond_features = 'bond_type', 'bond_stereo', 'is_edited'


@gin.configurable(blacklist=['n_atom_actions', 'n_bond_actions', 'prop2oh'])
class Megan(nn.Module):
    def __init__(self, n_atom_actions: int, n_bond_actions: int, prop2oh: dict,
                 bond_emb_dim: int = 8, hidden_dim: int = 512, stateful: bool = True,
                 atom_feature_keys: Tuple[str] = default_atom_features,
                 bond_feature_keys: Tuple[str] = default_bond_features,
                 reaction_type_given: bool = False, n_reaction_types: int = 10,
                 reaction_type_emb_dim: int = 16):
        super(Megan, self).__init__()
        self.prop2oh = prop2oh
        self.n_actions = n_atom_actions
        self.n_bond_actions = n_bond_actions
        self.bond_emb_dim = bond_emb_dim
        self.hidden_dim = hidden_dim
        self.stateful = stateful
        self.reaction_type_given = reaction_type_given

        total_atom_oh_len = sum(len(self.prop2oh['atom'][key]) + 1 for key in atom_feature_keys)
        total_bond_oh_len = sum(len(self.prop2oh['bond'][key]) + 1 for key in bond_feature_keys)

        self.numbered_atom_oh_keys = [(ORDERED_ATOM_OH_KEYS.index(key), key) for key in atom_feature_keys]
        self.numbered_bond_oh_keys = [(ORDERED_BOND_OH_KEYS.index(key), key) for key in bond_feature_keys]

        if reaction_type_given:
            assert reaction_type_emb_dim < hidden_dim
            assert reaction_type_emb_dim < bond_emb_dim
            self.reaction_type_embedding = nn.Embedding(n_reaction_types, reaction_type_emb_dim)
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim - reaction_type_emb_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim - reaction_type_emb_dim)
        else:
            self.reaction_type_embedding = None
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim)

        self.encoder = MeganEncoder(hidden_dim=hidden_dim, bond_emb_dim=bond_emb_dim)
        self.decoder = MeganDecoder(hidden_dim=hidden_dim, bond_emb_dim=bond_emb_dim,
                                    n_atom_actions=n_atom_actions, n_bond_actions=n_bond_actions)

    def _preprocess(self, x: dict) -> dict:
        oh_atom_feats = []
        for i, key in self.numbered_atom_oh_keys:
            oh_feat = to_one_hot(x['node_features'][:, :, i], dims=len(self.prop2oh['atom'][key]) + 1)
            oh_atom_feats.append(oh_feat)
        # noinspection PyTypeChecker
        atom_feats = torch.cat(oh_atom_feats, dim=-1)

        x['node_features'] = atom_feats

        # "node_adj_mask" has shape fo the adjacency matrix and puts 1 in every place where a bond is possible
        node_adj_mask = x['node_mask'].unsqueeze(-1)
        node_adj_mask = node_adj_mask.expand(*node_adj_mask.shape)
        node_adj_mask = node_adj_mask * node_adj_mask.permute(0, 2, 1, 3).contiguous()
        x['node_adj_mask'] = node_adj_mask

        conv_mask = x['adj_mask'].float().squeeze(-1)

        # two different masks are needed to mask for softmax activation
        conv_soft_mask = (-conv_mask + 1.0) * -1e9
        x['conv_mask'] = conv_mask
        x['conv_soft_mask'] = conv_soft_mask

        oh_bond_feats = []
        for i, key in self.numbered_bond_oh_keys:
            oh_feat = to_one_hot(x['adj'][:, :, :, i], dims=len(self.prop2oh['bond'][key]) + 1)
            oh_bond_feats.append(oh_feat)
        # noinspection PyTypeChecker
        adj = torch.cat(oh_bond_feats, dim=-1)
        x['adj'] = adj

        x['node_features'] = self.atom_embedding(x['node_features'])
        x['adj'] = self.bond_embedding(x['adj'])

        if self.reaction_type_given:
            reaction_type = x['reaction_type']
            r_type_emb = self.reaction_type_embedding(reaction_type)
            r_type_emb = r_type_emb.unsqueeze(1).expand(-1, x['node_features'].shape[1], -1)

            x['node_features'] = torch.cat((x['node_features'], r_type_emb), dim=-1)

            r_type_emb = r_type_emb.unsqueeze(2).expand(-1, -1, x['node_features'].shape[1], -1)
            x['adj'] = torch.cat((x['adj'], r_type_emb), dim=-1)

        return x

    def forward(self, x: dict) -> dict:
        batch_size, n_steps, n_nodes = x['adj'].shape[0], x['adj'].shape[1], x['adj'].shape[2]
        outputs = []
        output_masks = []
        state_dict = None

        for step_i in range(n_steps):
            step_batch = dict((k, v[:, step_i]) for k, v in x.items() if v.dim() > 1)

            step_results = self.forward_step(step_batch, state_dict=state_dict)
            state_dict = {
                'state': step_results['state'],
            }

            outputs.append(step_results['output'])
            output_masks.append(step_results['output_mask'])

        outputs = torch.stack(outputs, dim=1)
        output_masks = torch.stack(output_masks, dim=1)

        result = {
            'output': outputs,
            'output_mask': output_masks
        }

        return result

    def forward_step(self, step_batch: dict, state_dict=Optional[dict],
                     first_step: Optional[List[int]] = None) -> dict:
        batch_size, n_nodes = step_batch['adj'].shape[0], step_batch['adj'].shape[2]

        step_batch = self._preprocess(step_batch)

        # run encoder only on the first step of generation
        if self.stateful:
            if state_dict is None:  # first generation step
                step_batch = self.encoder(step_batch)
            else:  # generation step > 1
                state = state_dict['state']
                if state.shape[1] != n_nodes:
                    min_n_nodes = min(state.shape[1], n_nodes)
                    new_state = torch.zeros((batch_size, n_nodes, self.hidden_dim), device=device)
                    new_state[:, :min_n_nodes] = state[:, :min_n_nodes]
                    state = new_state

                # merge embeddings of nodes with their "state" (features taken from previous decoder)
                merged_node_features = torch.max(step_batch['node_features'], state)

                # this means there can be some samples for which this is the first step
                if first_step:
                    encoded_step_batch = self.encoder(step_batch)

                    # for samples for which this is not the first step, ignore encoder results
                    for i, first in enumerate(first_step):
                        if first:
                            step_batch['node_features'][i] = encoded_step_batch['node_features'][i]
                        else:
                            step_batch['node_features'][i] = merged_node_features[i]
                else:
                    step_batch['node_features'] = merged_node_features

            state, output, mask = self.decoder(step_batch)

            result = {
                'state': state,
                'output': output,
                'output_mask': mask
            }
        else:
            step_batch = self.encoder(step_batch)
            _, output, mask = self.decoder(step_batch)
            result = {
                'output': output,
                'output_mask': mask
            }
        return result
