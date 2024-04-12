import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv,HANConv
from torch_geometric.data import Data

class HAN(torch.nn.Module):
    """ Implementation of HAN mode l"""
    
    def __init__(self,label, in_channels: int=-1, hidden_channels: int=64,
              out_channels: int=2, n_heads=4, metadata=None,dropout = 0.5):
        super(HAN, self).__init__()
        self.label = label
        self.han_conv_0 = HANConv(in_channels,
                               hidden_channels,
                               heads=n_heads,
                               metadata=metadata,
                               dropout = dropout)
        self.han_conv_1 = HANConv(hidden_channels,
                               hidden_channels,
                               heads=n_heads,
                               metadata=metadata,
                               dropout = dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
 
    def forward(self, x_dict, edge_index_dict):
        
        x = self.han_conv_0(x_dict, edge_index_dict)
        x = self.lin(x[self.label])
        return x