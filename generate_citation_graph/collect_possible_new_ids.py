import requests
import os
from pathlib import Path
import pickle
import logging
from time import sleep
from tqdm import tqdm
import sys

class FlexibleCitationParser:
    def __init__(self, paper_records_path='data/teacher_graph/records/paper_records.pkl'):
        self.batch_size = 100
        self.citations_data = {}
        self.counter = 0
        # self.paper_records_path = paper_records_path
        # self.paper_records = pickle.load(open(self.paper_records_path, "rb"))
    
    def get_bulk_citations_from_id(self, ids, fields):
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
                response = requests.post(url, params=params, json={"ids": ids}, headers=headers)
                if response.status_code == 429:
                    raise Exception("Rate limit exceeded")
                elif response.status_code == 200:
                    break
                else:
                    logging.error(f"Error fetching citations for {ids}: {response.status_code}")
            except Exception as e:
                logging.error(f"Error fetching citations for {ids}: {e}")
                sleep(sleep_time)
                patience_limit -= 1
        if patience_limit == 0 and response.status_code != 200:
            logging.error(f"Error fetching citations for {ids}: {response.status_code}, s2 api is down")

        return response
    
    def save_processed_ids(self):
        pickle.dump(self.citations_data, open('data/teacher_graph/possible_new_data.pkl', "wb"))
        

    def collect_citations_batch(self, s2_ids, fields=None):
        """Collect citations in batches"""
            
        for paper_idx in tqdm(range(0, len(s2_ids), self.batch_size), total=len(s2_ids)//self.batch_size):
            ids = [i for i in s2_ids[paper_idx:paper_idx + self.batch_size]]

            info_with_references = self.get_bulk_citations_from_id(ids, fields)
            info_with_references = info_with_references.json()

            for i, paper_id in enumerate(s2_ids[paper_idx:paper_idx + self.batch_size]):
                info = info_with_references[i]
                if len(info['citations']) == 0:
                    logging.error(f"Error fetching citations for {paper_id}")
                    continue
                
                if info == None:
                    logging.error(f"Error fetching citations for {paper_id}")
                    continue

                self.counter += len(info['citations'])
                self.citations_data[paper_id] = info
            
            self.save_processed_ids()
            logging.info(f"Processed {paper_idx} papers")
            print(f"Processed {paper_idx} papers")
            print("Total citations collected: ", self.counter)
        

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

    # load the ids
    paper_ids = list(pickle.load(open('data/teacher_graph/paper_ids.pkl', "rb")))

    citation_parser = FlexibleCitationParser()
    fields = "paperId,citations.paperId,citations.venue"
    citation_parser.collect_citations_batch(s2_ids=paper_ids, fields=fields)   
    citation_parser.save_processed_ids()
    logging.info("Sleeping for 2 minutes...")
    sleep(120)

        
    
