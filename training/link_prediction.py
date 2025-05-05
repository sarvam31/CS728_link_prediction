import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GATConv
from torch.utils.data import DataLoader
from tqdm import tqdm

class PaperLinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rel_names):
        super().__init__()
        self.conv = HeteroGraphConv({rel: GATConv(in_dim, out_dim, num_heads=1) for rel in rel_names}, aggregate='mean')

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        return {k: v.squeeze(1) for k, v in h.items()}

class HGTModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, rel_names):
        super().__init__()
        self.layer1 = HGTLayer(in_dim, hidden_dim, rel_names)
        self.layer2 = HGTLayer(hidden_dim, hidden_dim, rel_names)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = self.layer2(g, h)
        return h

def compute_link_loss(paper_emb, edge_index):
    src, dst = edge_index
    src_emb = paper_emb[src]
    dst_emb = paper_emb[dst]
    scores = (src_emb * dst_emb).sum(dim=1)
    loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
    return loss

def evaluate(model, graph, features, eval_src_nodes, candidate_dst_nodes):
    model.eval()
    with torch.no_grad():
        h_dict = model(graph, features)
        paper_emb = h_dict['paper']
        src_emb = paper_emb[eval_src_nodes]
        dst_emb = paper_emb[candidate_dst_nodes]
        sim = torch.matmul(src_emb, dst_emb.T)  # shape: [eval_nodes, candidate_nodes]
        # Evaluation metric placeholder (e.g. recall@k)
        return sim

def train_model(graph, features, train_edges, eval_ids, epochs=10, lr=1e-3):
    model = HGTModel(in_dim=features['paper'].shape[1], hidden_dim=256, rel_names=graph.etypes)
    predictor = PaperLinkPredictor(256, 128, 256)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        h_dict = model(graph, features)
        paper_emb = predictor(h_dict['paper'])

        loss = compute_link_loss(paper_emb, train_edges)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Eval
        eval_sim = evaluate(model, graph, features, eval_ids, train_edges[0])
        print("Eval similarity shape:", eval_sim.shape)

    return model, predictor
