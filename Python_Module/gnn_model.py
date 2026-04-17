import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels=1, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.classifier = torch.nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)