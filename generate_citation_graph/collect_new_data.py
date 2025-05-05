import sys
import os

from collect_citations import CitationParser
import requests
from pathlib import Path
import pickle
import logging
from time import sleep
from tqdm import tqdm

        

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

    # TODO set initial paths
    citation_parser = CitationParser()
    fields = "paperId,citations.paperId,citations.venue,citations.year,citations.authors"
    # citation_parser.collect_citations_batch()
    while(True):
        citation_parser.collect_citations(fields=fields)   
        citation_parser.save_processed_ids()
        logging._ExcInfoType("Sleeping for 2 minutes...")
        sleep(120)
    
