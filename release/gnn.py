import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotLinkPredictor(nn.Module):
    def forward(self, h, src_idx, dst_idx):
        return (h[src_idx] * h[dst_idx]).sum(dim=-1)
    
    # Alternative implementation that avoids the in-place operation issue
class GATLinkPredictorFixed(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(GATLinkPredictorFixed, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, 1)
    
    def forward(self, g, features):
        # For NetworkX graph compatibility
        if isinstance(g, nx.DiGraph):
            edge_index = torch.tensor(list(g.edges())).t().to(device)
            # create empty tensor of size (2, num_edges)
            if edge_index.shape[0] == 0:
                edge_index = torch.zeros((2, len(10)), dtype=torch.long).to(device)
            h = self.gat1(features, edge_index)
            h = F.elu(h.flatten(1))
            h = self.gat2(h, edge_index).squeeze(1)
        else:
            # Original implementation for other graph types
            h = self.gat1(g, features)
            h = F.elu(h.flatten(1))
            h = self.gat2(g, h).squeeze(1)
        return h