import os
import sys
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
sys.path.insert(1, os.getenv("PROJECT_ROOT"))
os.environ['HF_HOME'] = os.getenv("HF_CACHE")

from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
import dgl
from dgl import heterograph
from collections import defaultdict
import numpy as np

# Load SciBERT model
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", device_map="cuda:0", local_files_only=True)
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", device_map="cuda:0", local_files_only=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_scibert_embedding(title, abstract):
    text = title + "\n" + abstract
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding  # shape: (1, hidden_size)

paper_records = pickle.load(open('data/teacher_graph/records/paper_records.pkl', "rb"))
author_records = pickle.load(open('data/teacher_graph/records/author_records.pkl', "rb"))
fields_of_study = set()
affiliations = set()

for author_id, author in tqdm(author_records.items()):
    affiliations.update(author['affiliations'])

for s2_id, paper in tqdm(paper_records.items()):
    fields_of_study.update(paper['field_ids'])

affiliations = [{i: affiliation} for i, affiliation in enumerate(affiliations)]
fields_of_study = [{i: field} for i, field in enumerate(fields_of_study)]

all_ids = set(paper_records.keys())
student_paper_ids = pickle.load(open('data/teacher_graph/paper_ids.pkl', "rb"))
eval_ids = pickle.load(open('data/teacher_graph/eval_ids.pickle', "rb"))
test_ids = pickle.load(open('data/teacher_graph/test_ids.pickle', "rb"))
extra_train_ids = pickle.load(open('data/teacher_graph/extra_train_ids.pickle', "rb"))

# create a combined set of student paper ids and extra train ids
train_ids = list(set(student_paper_ids).union(extra_train_ids))

# First, index all node types
paper_id_map = {pid: i for i, pid in enumerate(train_ids)}
author_id_map = {aid: i for i, aid in enumerate(author_records.keys())}
field_id_map = {f[next(iter(f))]: i for i, f in enumerate(fields_of_study)}
affil_id_map = {a[next(iter(a))]: i for i, a in enumerate(affiliations)}

# Reverse lookup for embedding later
inv_paper_id_map = {v: k for k, v in paper_id_map.items()}

# Containers for edges
edge_dict = defaultdict(list)

# Paper -> Paper (Citation)
for pid in tqdm(train_ids):
    pdata = paper_records[pid]
    for ref_id in pdata['reference_ids']:
        if ref_id is not None and ref_id in train_ids:
            edge_dict[('paper', 'cites', 'paper')].append((paper_id_map[pid], paper_id_map[ref_id]))

# Paper -> Author
for pid in tqdm(train_ids):
    pdata = paper_records[pid]
    for aid in pdata['author_ids']:
        if aid is not None and aid in author_id_map:
            edge_dict[('paper', 'written_by', 'author')].append((paper_id_map[pid], author_id_map[aid]))

# Paper -> Field of Study
for pid in tqdm(train_ids):
    pdata = paper_records[pid]
    for fid in pdata['field_ids']:
        if fid is not None and fid in field_id_map:
            edge_dict[('paper', 'has_field', 'field_of_study')].append((paper_id_map[pid], field_id_map[fid]))

# Author -> Institute
for aid, adata in tqdm(author_records.items()):
    for affil in adata['affiliations']:
        if affil in affil_id_map:
            edge_dict[('author', 'affiliated_with', 'institute')].append((author_id_map[aid], affil_id_map[affil]))

# Node counts
num_paper = len(paper_id_map)
num_author = len(author_id_map)
num_field = len(field_id_map)
num_institute = len(affil_id_map)

# Create heterograph
g = dgl.heterograph(edge_dict,
    num_nodes_dict={
        'paper': num_paper,
        'author': num_author,
        'field_of_study': num_field,
        'institute': num_institute
    }
)

# Compute and assign SciBERT embeddings for paper nodes
print("Generating SciBERT embeddings...")
paper_feats = torch.zeros((num_paper, model.config.hidden_size))

for local_idx, global_id in tqdm(inv_paper_id_map.items()):
    pdata = paper_records[global_id]
    emb = get_scibert_embedding(pdata['title'], pdata['abstract'])
    paper_feats[local_idx] = emb

g.nodes['paper'].data['feat'] = paper_feats

# (Optional) Assign other metadata as features (venue, year, etc.) later as needed

print("Graph creation complete with node types:", g.ntypes)
print("Edge types:", g.etypes)
print("Graph summary:")
print(g)


