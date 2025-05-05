from dataclasses import dataclass
from typing import List
import re
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec
from pathlib import Path
from tqdm import tqdm
import os
import pickle
import bibtexparser

num_blocks = set()

@dataclass
class BibEntry:
    """Represents a bibliography entry"""
    bibitem_text: str  # Text after \bibitem until first \newblock
    blocks: List[str]  # List of text blocks after each \newblock

# def create_latex_converter():
#     """Create a custom LaTeX converter with additional command handling"""
#     # Create custom specs for handling special LaTeX commands
#     specs = {
#         'href': MacroTextSpec('href', '{url}{text}', '\\href{%(url)s}{%(text)s}'),
#         'url': MacroTextSpec('url', '{text}', '%(text)s'),
#         'texttt': MacroTextSpec('texttt', '{text}', '%(text)s'),
#         'emph': MacroTextSpec('emph', '{text}', '%(text)s'),
#         'em': MacroTextSpec('em', '{text}', '%(text)s'),
#     }
    
#     return LatexNodes2Text(math_mode='text', macro_list=list(specs.values()))

def clean_latex_commands(text: str) -> str:
    """Clean LaTeX commands before conversion - ONLY handle \href"""
    # Updated pattern for \href with possible spaces between elements
    text = re.sub(r'\\href\s*{([^}]*)}\s*{([^}]*)}', r'\2', text)
    return text


def parse_bbl_file(file_path: str) -> List[BibEntry]:
    """Parse a BBL file and extract bibliography entries with all blocks"""
    entries = []

    with open(file_path, 'r', encoding='iso-8859-1') as f:
        content = f.read()

    # Remove LaTeX comments and normalize whitespace
    content = re.sub(r'%.*?\n', '\n', content)
    content = re.sub(r'\s+', ' ', content)

    # Find all bibliography entries
    pattern = r'\\bibitem(?:\[.*?\])*{.*?}(.*?)(?=\\bibitem|\\end{thebibliography})'
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        entry_text = match.group(1).strip()
        
        # Split on \newblock and clean each block
        raw_blocks = entry_text.split(r'\newblock')
        blocks = []
        for block in raw_blocks:
            # First clean LaTeX commands
            cleaned = clean_latex_commands(block.strip())
            # Then convert remaining LaTeX
            cleaned = LatexNodes2Text().latex_to_text(cleaned)
            blocks.append(cleaned.strip())
        num_blocks.add(len(blocks))
        # blocks = [b.strip() for b in entry_text.split(r'\newblock')]
        
        # First part is bibitem content, rest are newblocks
        bibitem_text = blocks[0]
        newblocks = blocks[1] if len(blocks) > 1 else []
        
        entries.append(newblocks)
    
    return entries

def parse_bib_file(bib_path, source_id=None):
    """Parse citations from .bib file"""
    titles = []
    try:
        with open(bib_path, 'r', encoding='iso-8859-1') as f:
            bib_database = bibtexparser.load(f)

        for entry in bib_database.entries:
            if 'title' in entry:
                titles.append(entry['title'])
                
    except Exception as e:
        print(f"Error parsing {bib_path}: {e}")
    
    return titles

def extract_citation_info(entry: BibEntry) -> dict:
    """Extract structured information from a bibliography entry"""
    info = {
        'authors': [],
        'year': None,
        'title': None,
        'venue': None,
        'additional_info': []
    }
    
    # Extract authors from bibitem text
    author_line = entry.bibitem_text.split('\n')[0]
    author_parts = re.split(r',\s*&\s*|\s*&\s*|,\s*and\s*|,\s*', author_line)
    info['authors'] = [a.strip() for a in author_parts if a.strip()]
    
    # Extract year from bibitem text or first block
    year_match = re.search(r'\b(19|20)\d{2}[a-z]?\b', entry.bibitem_text)
    if year_match:
        info['year'] = year_match.group(0)
    
    # Process blocks
    if entry.blocks:
        # First block usually contains title
        info['title'] = entry.blocks[0].strip()
        
        # Second block usually contains venue
        if len(entry.blocks) > 1:
            info['venue'] = entry.blocks[1].strip()
        
        # Store any additional blocks
        if len(entry.blocks) > 2:
            info['additional_info'] = [b.strip() for b in entry.blocks[2:]]
    
    return info

if __name__ == "__main__":
    
    # recursively find all .bbl files
    bbl_files = []

    for path in tqdm(Path('data/dataset_papers').rglob('*.bbl')):
        bbl_files.append(path)
    
    # load title_to_id pickle from the file
    title_to_id = pickle.load(open("data/title_to_arxiv_id.pickle", "rb"))

    for i in bbl_files:
        # list all files in the directory
        directory = f"data/dataset_papers/{i}"
        files = os.listdir(directory)
        for f in files:
            if ".bbl" in f:
                f = os.path.join(directory, f)
                bbl_files.append(f)
        # iterate over all the files in the directory
    print(bbl_files)
    another_map = {}
    for bbl_file in tqdm(bbl_files):
        entries = parse_bbl_file(bbl_file)
        num_blocks.add(len(entries))
        print(entries)
        print(f"Found {len(entries)} entries in {bbl_file}")
        print("\n\n")
        for k in special_papers:
            if k in bbl_file:
                if k not in another_map:
                    another_map[k] = []
                for entry in entries:
                    if entry in title_to_id:
                        ref_arxiv_id = title_to_id[entry]
                        another_map[k].append(ref_arxiv_id)

    for bib_file in tqdm(bib_files):
        entries = parse_bib_file(bib_file)
        num_blocks.add(len(entries))
        print(entries)
        print(f"Found {len(entries)} entries in {bib_file}")
        print("\n\n")
        for k in special_papers:
            if k in bib_file:
                if k not in another_map:
                    another_map[k] = []
                for entry in entries:
                    if entry in title_to_id:
                        ref_arxiv_id = title_to_id[entry]
                        another_map[k].append(ref_arxiv_id)

