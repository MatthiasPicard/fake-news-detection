

"Test of another graph neural network model"

import torch
from torch_geometric.nn import GCNConv

class convGNN(torch.nn.Module):
    
    def __init__(self,metadata,hidden_channels,out_channels,n_heads,dropout=None):
        super(convGNN, self).__init__()
        
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.conv1 = GCNConv(self.metadata.num_features, self.hidden_channels)
        self.lin = torch.nn.Linear(self.hidden_channels, self.out_channels)
               

    def forward(self, x_dict, edge_index_dict):
    
        x = self.conv1(x_dict[self.metadata.label], edge_index_dict[self.metadata.label])
        # print(x)
        # x = x.relu()
        # x = self.han_conv_1(x, edge_index_dict)
        x = self.lin(x[self.label])
        return x
    