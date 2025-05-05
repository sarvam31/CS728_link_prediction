import requests
import os
from pathlib import Path
import pickle
import logging
from time import sleep
from tqdm import tqdm
import sys

class CitationParser:
    def __init__(self, processed_ids_file='data/processed_ids.pickle', papers_data_file='data/papers_data.pickle', s2_map_file='data/s2_map.pickle'):
        self.paper_index = pickle.load(open(papers_data_file, "rb")) if os.path.exists(papers_data_file) else {}
        self.processed = pickle.load(open(processed_ids_file, "rb")) if os.path.exists(processed_ids_file) else []
        self.s2_map = pickle.load(open(s2_map_file, "rb")) if os.path.exists(s2_map_file) else {}
        self.batch_size = 100

        self.processed_path='data/processed_ids.pickle'
        self.paper_index_path='data/papers_data.pickle'
        self.s2_map_path='data/s2_map.pickle'

        self.citations_count = 0

    def extract_paper_info(self, folder_path, base_path = Path('data/dataset_papers/')):
            """Extract basic paper info from a paper folder"""
            paper_id = Path(folder_path).name

            # Check if the paper ID is already processed, which happens when repeat runs are done due to api issues
            if paper_id in self.paper_index:
                return paper_id.split('v')[0]
            
            # Get title
            title_path = base_path / paper_id / "title.txt"
            if title_path.exists():
                with open(title_path, 'r', encoding='utf-8') as f:
                    title = f.read().strip()
            else:
                raise ValueError("Title not found")
                    
            # Get abstract
            abstract_path = base_path / paper_id / "abstract.txt"
            abstract = ""
            if abstract_path.exists():
                with open(abstract_path, 'r', encoding='utf-8') as f:
                    abstract = f.read().strip()
            else:
                raise ValueError("Abstract not found")
                    
            self.paper_index[paper_id] = {
                'arxiv_id': paper_id,
                'title': title,
                'abstract': abstract,
            }
            return paper_id.split('v')[0]  # Return the arXiv ID without version

    def get_citations_from_arxiv_id(self, arxiv_id, fields):
        """Get citations from arXiv ID"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        params = {
            "fields": fields
        }
        headers = {"User-Agent": "CitationBot/1.0"}
        sleep(1)  # Sleep for 1 second to avoid hitting the API too fast
        patience_limit = 20
        sleep_time = 2
        while(patience_limit > 0):
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    raise Exception("Rate limit exceeded")
                elif response.status_code == 200:
                    break
                else:
                    logging.error(f"Error fetching citations for {arxiv_id}: {response.status_code}")
            except Exception as e:
                # print(f"Error fetching citations for {arxiv_id}: {e}")
                logging.error(f"Error fetching citations for {arxiv_id}: {e}")
                sleep(sleep_time)
                patience_limit -= 1
        if patience_limit == 0 and response.status_code != 200:
            logging.error(f"Error fetching citations for {arxiv_id}: {response.status_code}, s2 api is down")

        return response
    
    def get_bulk_citations_from_arxiv_id(self, arxiv_ids, fields):
        """Get citations from a list of arXiv IDs using batch API"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {
            "fields": fields
        }
        headers = {"User-Agent": "CitationBot/1.0"}
        sleep(1)  # Sleep for 1 second to avoid hitting the API too fast
        patience_limit = 20
        sleep_time = 2
        while(patience_limit > 0):
            try:
                response = requests.post(url, params=params, json={"ids": arxiv_ids}, headers=headers)
                if response.status_code == 429:
                    raise Exception("Rate limit exceeded")
                elif response.status_code == 200:
                    break
                else:
                    logging.error(f"Error fetching citations for {arxiv_ids}: {response.status_code}")
            except Exception as e:
                logging.error(f"Error fetching citations for {arxiv_ids}: {e}")
                sleep(sleep_time)
                patience_limit -= 1
        if patience_limit == 0 and response.status_code != 200:
            logging.error(f"Error fetching citations for {arxiv_ids}: {response.status_code}, s2 api is down")

        return response
    
    def save_processed_ids(self):
        pickle.dump(self.processed, open(self.processed_path, "wb"))
        pickle.dump(self.paper_index, open(self.paper_index_path, "wb"))
        pickle.dump(self.s2_map, open(self.s2_map_path, "wb"))
        
    def collect_citations(self, data_dir='data/dataset_papers', fields=None):
        folder_names = [i.name for i in Path(data_dir).glob("*/")]

        # # randomly select 100 papers to process
        # import random
        # random.seed(42)
        # folder_names = random.sample(folder_names, 100)

        if set(folder_names) == set(self.processed):
            logging.info("All papers have already been processed.")
            sys.exit(0)

        for idx, paper_dir in enumerate(Path(data_dir).glob("*/")):

            if len(self.processed) % 20 == 0:
                print(f"Processed {len(self.processed)} papers")
                print("Citations count: ", self.citations_count)
                self.save_processed_ids()

            # Skip if already processed
            if paper_dir.name in self.processed:
                continue
            
            # Extract paper info
            paper_id = paper_dir.name
            if paper_id not in self.paper_index:
                self.extract_paper_info(paper_dir)
            info_with_references = self.get_citations_from_arxiv_id(arxiv_id=paper_id.split('v')[0], fields=fields)  # Alon & Yahav paper
            
            if info_with_references.status_code != 200:
                logging.error(f"Error fetching citations for {paper_id}: {info_with_references.status_code}")
                continue
            info_with_references = info_with_references.json()
            info_with_references['s2_paperId'] = info_with_references.pop('paperId', None)
            # Update paper_index with all fields from info_with_references
            for key, value in info_with_references.items():
                self.paper_index[paper_id][key] = value
            
            # Process the response
            self.processed.append(paper_dir.name)

            self.s2_map[info_with_references.get("s2_paperId")] = paper_id

            self.citations_count += len(info_with_references["citations"])

            # if info_with_references.get("citationCount") > 9990:
            #     logging.warning(f"Paper {paper_id} has too many citations: {info_with_references.get('citationCount')}")

    def collect_citations_batch(self, data_dir='data/dataset_papers', fields=None):
        """Collect citations in batches"""
        folder_names = [i.name for i in Path(data_dir).glob("*/")]

        if set(folder_names) == set(self.processed):
            logging.info("All papers have already been processed.")
            sys.exit(0)

        for idx, paper_dir in tqdm(enumerate(folder_names), total=len(folder_names)):
            if paper_dir in self.processed:
                continue
            self.extract_paper_info(paper_dir)

        self.save_processed_ids()
            
        for paper_idx in tqdm(range(0, len(folder_names), self.batch_size), total=len(folder_names)//self.batch_size):
            ids = ['ARXIV:' + i.split('v')[0] for i in folder_names[paper_idx:paper_idx + self.batch_size]]

            if all([i in self.processed for i in folder_names[paper_idx:paper_idx + self.batch_size]]):
                print(f"Processed {paper_idx} papers")
                continue

            info_with_references = self.get_bulk_citations_from_arxiv_id(ids, fields)
            info_with_references = info_with_references.json()

            self.processed.extend(folder_names[paper_idx:paper_idx + self.batch_size])

            for i, paper_id in enumerate(folder_names[paper_idx:paper_idx + self.batch_size]):
                info = info_with_references[i]
                
                if info == None:
                    logging.error(f"Error fetching citations for {paper_id}: {self.paper_index[paper_id]}")
                    continue
                
                if info.get("citationCount") > 9990:
                    print(f"Paper {paper_id} has too many citations: {info.get('citationCount')}")
                    logging.warning(f"Paper {paper_id} has too many citations: {info.get('citationCount')}")
                    continue

                info['s2_paperId'] = info.pop('paperId', None)
                # Update paper_index with all fields from info
                for key, value in info.items():
                    self.paper_index[paper_id][key] = value

                self.s2_map[info.get("s2_paperId")] = paper_id
            
            self.save_processed_ids()
            logging.info(f"Processed {paper_idx} papers")
            print(f"Processed {paper_idx} papers")
        

if __name__ == "__main__":
    # get todays date in format DDMMYYYY
    import datetime
    TODAYS_DATE = datetime.datetime.now().strftime("%d%m%Y")
    TIMENOW = datetime.datetime.now().strftime("%H%M%S")

    log_file_path = f'logs/collect_citations_data/{TODAYS_DATE}_{TIMENOW}.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        filename=log_file_path, 
        format='%(levelname)s:%(message)s',
        filemode='w',
        level=logging.INFO
    )

    citation_parser = CitationParser()
    fields = "paperId,corpusId,externalIds,url,title,abstract,venue,year,citationCount,influentialCitationCount,referenceCount,fieldsOfStudy,publicationTypes,authors.authorId,authors.name,authors.externalIds,authors.affiliations,authors.paperCount,authors.citationCount,authors.hIndex,authors.url,authors.homepage,references.paperId,references.title,references.abstract,references.venue,references.year,references.citationCount,references.referenceCount,references.fieldsOfStudy,references.influentialCitationCount,references.authors"
    while(True):
        citation_parser.collect_citations_batch(fields=fields)   
        # citation_parser.collect_citations()
        citation_parser.save_processed_ids()
        logging.info("Sleeping for 2 minutes...")
        sleep(120)
    
