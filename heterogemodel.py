import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
from torch_geometric.data import Data
import torch.nn as nn


class heterognn(torch.nn.Module):
    def __init__(self, in_channels: int = 300, hidden_channels: int = 300,
                 out_channels: int = 300, n_heads=8, metadata=None):
        super(heterognn, self).__init__()
        torch.manual_seed(12345)
        self.han_conv_0 = torch_geometric.nn.HANConv(in_channels=in_channels,
                                                     hidden_channels=in_channels,
                                                     out_channels=hidden_channels,
                                                     heads=n_heads,
                                                     metadata=metadata)
        if out_channels:
            input_lin_dim = out_channels
        else:
            input_lin_dim = hidden_channels
        self.lin = torch.nn.Linear(input_lin_dim, 2)

    def forward(self, x_dict, edge_index_dict):
        # 1. Obtain node embeddings
        x = self.han_conv_0(x_dict, edge_index_dict)
        # 2. Readout layer
        x = torch_geometric.nn.pool.global_mean_pool(x["openie"])  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x