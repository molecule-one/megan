"""
Decoder of MEGAN model.
Outputs final action probabilities for atoms and bonds
"""
from typing import Tuple

import gin
import torch
from torch import nn

from src.model.graph.gat import MultiHeadGraphConvLayer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def softmax(values, base, dim):
    exp = torch.exp(base * values)
    return exp / torch.sum(exp, dim=dim).unsqueeze(-1)


@gin.configurable(blacklist=['hidden_dim', 'bond_emb_dim', 'n_atom_actions', 'n_bond_actions'])
class MeganDecoder(nn.Module):
    def __init__(self, hidden_dim: int, bond_emb_dim, n_atom_actions: int, n_bond_actions: int,
                 n_fc: int = 3, n_decoder_conv: int = 4, dec_residual: bool = True, bond_atom_dim: int = 32,
                 atom_fc_hidden_dim: int = 256, bond_fc_hidden_dim: int = 256, dec_dropout: float = 0.0,
                 dec_hidden_dim: int = 0, dec_att_heads: int = 0):
        super(MeganDecoder, self).__init__()
        if dec_hidden_dim == 0:
            dec_hidden_dim = hidden_dim

        self.n_actions = n_atom_actions
        self.n_bond_actions = n_bond_actions
        self.hidden_dim = hidden_dim
        self.n_fc = n_fc
        self.n_conv = n_decoder_conv
        self.residual = dec_residual
        self.atom_fc_hidden_dim = atom_fc_hidden_dim
        self.bond_fc_hidden_dim = bond_fc_hidden_dim
        self.bond_atom_dim = bond_atom_dim
        self.dropout = nn.Dropout(dec_dropout) if dec_dropout > 0 else lambda x: x

        self.fc_atom_layers = []
        self.fc_bond_layers = []

        self.conv_layers = []
        for i in range(self.n_conv):
            input_dim = hidden_dim if i == 0 else dec_hidden_dim
            output_dim = hidden_dim if i == self.n_conv - 1 else dec_hidden_dim
            if dec_att_heads == 0:
                conv = MultiHeadGraphConvLayer(bond_dim=bond_emb_dim, input_dim=input_dim, output_dim=output_dim,
                                               residual=False)
            else:
                conv = MultiHeadGraphConvLayer(bond_dim=bond_emb_dim, input_dim=input_dim, output_dim=output_dim,
                                               residual=False, att_heads=dec_att_heads)

            self.conv_layers.append(conv)
            setattr(self, f'MultiHeadGraphConv_{i + 1}', conv)

        for i in range(n_fc):
            in_dim = hidden_dim if i == 0 else atom_fc_hidden_dim
            out_dim = atom_fc_hidden_dim if i < n_fc - 1 else n_atom_actions

            atom_fc = nn.Linear(in_dim, out_dim)
            setattr(self, f'fc_atom_{i + 1}', atom_fc)
            self.fc_atom_layers.append(atom_fc)

        self.fc_atom_bond = nn.Linear(hidden_dim, bond_atom_dim)

        for i in range(n_fc):
            in_dim = bond_atom_dim + bond_emb_dim if i == 0 else bond_fc_hidden_dim
            out_dim = bond_fc_hidden_dim if i < n_fc - 1 else n_bond_actions

            bond_fc = nn.Linear(in_dim, out_dim)
            setattr(self, f'fc_bond_{i + 1}', bond_fc)
            self.fc_bond_layers.append(bond_fc)

    def _forward_atom_features(self, atom_feats):
        for layer in self.fc_atom_layers[:-1]:
            atom_feats = torch.relu(layer(atom_feats))
            atom_feats = self.dropout(atom_feats)
        return atom_feats

    def _forward_bond_features(self, atom_feats, adj):
        atom_feats = torch.relu(self.fc_atom_bond(atom_feats))

        x_exp_shape = adj.shape[:-1] + (atom_feats.shape[-1],)
        x_rows = torch.unsqueeze(atom_feats, 1).expand(x_exp_shape)
        x_cols = torch.unsqueeze(atom_feats, 2).expand(x_exp_shape)

        x_sum = x_rows + x_cols
        bond_actions_feat = torch.cat([x_sum, adj], dim=-1)
        for bond_layer in self.fc_bond_layers[:-1]:
            bond_actions_feat = torch.relu(bond_layer(bond_actions_feat))
            bond_actions_feat = self.dropout(bond_actions_feat)

        return bond_actions_feat

    def forward_embedding(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_feats = x['node_features']
        prev_atom_feats = atom_feats

        for i, conv in enumerate(self.conv_layers):
            residual = self.residual and i % 2 == 1
            atom_feats = conv(atom_feats, x['adj'], x['conv_mask'], x['conv_soft_mask'],
                              apply_activation=not residual)
            atom_feats = self.dropout(atom_feats)
            if residual:
                atom_feats = torch.relu(atom_feats + prev_atom_feats)
                prev_atom_feats = atom_feats

        atom_feats = atom_feats * x['node_mask'].expand(*atom_feats.shape)
        node_state = atom_feats

        # calculate final features for atom and bond actions
        atom_actions_feat = self._forward_atom_features(atom_feats)
        bond_actions_feat = self._forward_bond_features(atom_feats, x['adj'])

        return node_state, atom_actions_feat, bond_actions_feat

    def forward(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_state, atom_actions_feat, bond_actions_feat = self.forward_embedding(x)

        atom_actions_feat = self.fc_atom_layers[-1](atom_actions_feat)
        bond_actions_feat = self.fc_bond_layers[-1](bond_actions_feat)

        max_feat_shape = max(atom_actions_feat.shape[-1], bond_actions_feat.shape[-1])

        atom_actions_exp = torch.zeros(atom_actions_feat.shape[:-1] + (max_feat_shape,), dtype=torch.float,
                                       device=device)
        atom_actions_exp[:, :, :atom_actions_feat.shape[-1]] = atom_actions_feat

        bond_actions_exp = torch.zeros(bond_actions_feat.shape[:-1] + (max_feat_shape,), dtype=torch.float,
                                       device=device)
        bond_actions_exp[:, :, :, :bond_actions_feat.shape[-1]] = bond_actions_feat

        bond_mask = x['node_adj_mask'].contiguous()
        bond_mask_exp = torch.zeros(bond_mask.shape[:-1] + (max_feat_shape,),
                                    dtype=torch.float, device=device)
        bond_mask = bond_mask * x['bond_action_mask']
        bond_mask_exp[:, :, :, :bond_actions_feat.shape[-1]] = bond_mask

        atom_mask = x['node_mask'].expand(*atom_actions_feat.shape)
        atom_mask = atom_mask * x['atom_action_mask']
        atom_mask_exp = torch.zeros(atom_mask.shape[:-1] + (max_feat_shape,),
                                    dtype=torch.float, device=device)
        atom_mask_exp[:, :, :atom_actions_feat.shape[-1]] = atom_mask

        atom_mask_exp = atom_mask_exp.unsqueeze(1)
        atom_actions_exp = atom_actions_exp.unsqueeze(1)

        all_actions = torch.cat([bond_actions_exp, atom_actions_exp], dim=1)
        all_actions_mask = torch.cat([bond_mask_exp, atom_mask_exp], dim=1)
        all_actions = torch.reshape(all_actions, (all_actions.shape[0], -1))
        all_actions_mask = torch.reshape(all_actions_mask, (all_actions_mask.shape[0], -1))

        soft_mask = (1.0 - all_actions_mask) * -1e9
        base = x.get('base', 1.0)

        if 'sigmoid' in x and x['sigmoid']:
            all_actions = torch.sigmoid(all_actions) * all_actions_mask
        else:
            if base == 1.0:
                all_actions = torch.softmax(all_actions + soft_mask, dim=-1) * all_actions_mask
            else:
                all_actions = softmax(all_actions + soft_mask, base=base, dim=-1) * all_actions_mask

        return node_state, all_actions, all_actions_mask
