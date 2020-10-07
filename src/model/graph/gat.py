# -*- coding: utf-8 -*-
"""
Custom implementation of a Graph Attention Network layer
"""
import logging

import gin
import torch
from torch import nn

logger = logging.getLogger(__name__)


@gin.configurable(whitelist=['att_heads', 'att_dim'])
class MultiHeadGraphConvLayer(nn.Module):
    def __init__(self, bond_dim: int, input_dim: int, output_dim: int, residual: bool,
                 att_heads: int = 8, att_dim: int = 64):
        """
        :param att_dim: dimensionality of narrowed nodes representation for the attention
        :param att_heads: number of attention heads
        """
        super(MultiHeadGraphConvLayer, self).__init__()
        self.n_att = att_heads
        self.att_dim = att_dim

        self.atoms_att = nn.Linear(input_dim, att_dim)
        self.final_att = nn.Linear(att_dim * 2 + bond_dim, att_heads)

        self.conv_layers = []

        if output_dim % att_heads != 0:
            raise ValueError(f"Output dimension ({output_dim} "
                             f"must be a multiple of number of attention heads ({att_heads}")

        for i in range(att_heads):
            conv = nn.Linear(input_dim, int(output_dim / att_heads))
            self.conv_layers.append(conv)
            setattr(self, f'graph_conv_{i + 1}', conv)

        self.residual = residual

    def forward(self, x, adj, mask, soft_mask, apply_activation: bool = True):
        x_att = torch.relu(self.atoms_att(x))
        x_att_shape = adj.shape[:-1] + (x_att.shape[-1],)
        x_rows = torch.unsqueeze(x_att, 1).expand(x_att_shape)
        x_cols = torch.unsqueeze(x_att, 2).expand(x_att_shape)

        x_att = torch.cat([x_rows, x_cols, adj], dim=-1)
        x_att = self.final_att(x_att)

        head_outs = []

        for i, conv in enumerate(self.conv_layers):
            att = x_att[:, :, :, i]
            att = torch.softmax(att + soft_mask, dim=-1) * mask
            out = torch.bmm(att, x)
            out = conv(out)
            head_outs.append(out)

        out = torch.cat(head_outs, dim=-1)

        if self.residual:
            out = x + out

        if apply_activation:
            out = torch.relu(out)

        return out
