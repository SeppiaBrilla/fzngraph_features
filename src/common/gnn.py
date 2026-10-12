import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class CustomGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGNN, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)
        self.layer3 = GCNConv(hidden_dim, output_dim)
        self.layer4 = GCNConv(hidden_dim, output_dim)

    def forward(self, feature_data, edge_info):
        # First GCN layer
        x = self.layer1(feature_data, edge_info)
        x = F.gelu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        # Second GCN layer
        x = self.layer2(x, edge_info)
        x = F.gelu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        # Third GCN layer
        x = self.layer3(x, edge_info)
        x = F.gelu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        # Fourth GCN layer
        x = self.layer4(x, edge_info)
        return F.log_softmax(x, dim=1)

