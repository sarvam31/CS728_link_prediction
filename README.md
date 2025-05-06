# CS728_link_prediction
Objective is to predict links from test set papers to train set papers

## Environment Setup
Follow these steps:
1. Clone the repository
2. conda create -n env_lwg python=3.12
3. Activate the environment
4. pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
5. pip install -r requirements.txt

## Evaluation
1. Ensure you are in project directory
2. Run evaluation using run_evaluation.py (Check for python/python3 in case of any error)
3. SciBERT model will be downloaded initially
4. Ensure "gat_model.pt", "node_id_to_arxiv_id.pkl" and "train_output.pkl" are present in release directory

## Directory Structure
1. Misc: Contains file to download initial data from google drive
2. Model: Saved model.pt files
3. Release: Files needed for evaluation
4. Training: Python files and notebooks used for training models following differnet approaches mentioend in report
5. Generate_citation_graph: Contains files to collect paper, authors data using semantic scholar api, parsing .bbl and .bib files and notebook to answers for part1 