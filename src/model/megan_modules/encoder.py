"""
Encoder of MEGAN model.
For a stateless model it is used for every sample.
For a stateful model is used only for the first sample of a reaction.
"""
import gin
import torch
from torch import nn

from src.model.graph.gat import MultiHeadGraphConvLayer


@gin.configurable(blacklist=['hidden_dim', 'bond_emb_dim'])
class MeganEncoder(nn.Module):
    def __init__(self, hidden_dim: int, bond_emb_dim: int, n_encoder_conv: int = 4,
                 enc_residual: bool = True, enc_dropout: float = 0.0):
        super(MeganEncoder, self).__init__()
        self.n_conv = n_encoder_conv
        self.residual = enc_residual
        self.dropout = nn.Dropout(enc_dropout) if enc_dropout > 0 else lambda x: x

        self.conv_layers = []
        for i in range(self.n_conv):
            conv = MultiHeadGraphConvLayer(bond_dim=bond_emb_dim, input_dim=hidden_dim,
                                           output_dim=hidden_dim, residual=False)
            self.conv_layers.append(conv)
            setattr(self, f'MultiHeadGraphConv_{i + 1}', conv)

    def forward(self, x: dict) -> dict:
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
        x['node_features'] = atom_feats
        return x
