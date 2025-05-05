import argparse

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################
    import os
    # os.environ['HF_HOME'] = os.getenv("HF_CACHE")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    import networkx as nx
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from pathlib import Path
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

    node_id_to_arxiv_id_path = Path("release/node_id_to_arxiv_id.pkl")
    node_id_to_arxiv_id = pickle.load(open(node_id_to_arxiv_id_path, "rb"))


    def get_scibert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding  # shape: (1, hidden_size)



    class DotLinkPredictor(nn.Module):
        def forward(self, h, src_idx, dst_idx):
            return (h[src_idx] * h[dst_idx]).sum(dim=-1)
        
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
                    edge_index = torch.zeros((2, 10), dtype=torch.long).to(device)
                h = self.gat1(features, edge_index)
                h = F.elu(h.flatten(1))
                h = self.gat2(h, edge_index).squeeze(1)
            else:
                # Original implementation for other graph types
                h = self.gat1(g, features)
                h = F.elu(h.flatten(1))
                h = self.gat2(g, h).squeeze(1)
            return h

    title = args.test_paper_title
    abstract = args.test_paper_abstract
    text = title + "\n" + abstract + "\n"
    eval_feats = get_scibert_embedding(text)

    eval_graph = nx.DiGraph()
    # add nodes for all papers
    eval_graph.add_nodes_from([0])
    # add edges for all papers
    edges = []
    eval_graph.add_edges_from(edges)

    in_dim = 768
    hidden_dim = 128
    num_heads = 4

    gnn_model = GATLinkPredictorFixed(in_dim, hidden_dim, num_heads)
    gnn_model.load_state_dict(torch.load('release/gat_model.pt'))
    gnn_model.to(device)  # move to GPU if needed
    gnn_model.eval()      # set to evaluation mode  
    eval_embs = gnn_model(eval_graph, eval_feats)

    train_output_path = Path('release/train_output.pkl')
    train_output = pickle.load(open(train_output_path, "rb"))
    train_output = torch.tensor(train_output).to(device)

    # Compute similarity scores
    scores = torch.matmul(eval_embs, train_output.T)
    # sort by scores in descending order and get top indices
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_indices = sorted_indices.cpu().numpy()[0]

    result = [node_id_to_arxiv_id[i] for i in sorted_indices]

    # # prepare a ranked list of papers like this:
    # result = ['paper1', 'paper2', 'paper3', 'paperK']  # Replace with your actual ranked list


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()