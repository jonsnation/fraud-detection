# gnn/model.py

from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch

class FraudGraphSAGE(nn.Module):  # <-- Must match name when saved
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return torch.sigmoid(self.classifier(x))
