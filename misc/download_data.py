# download data from drive url and save to local path then unzip

import os
import requests
import zipfile
import tarfile
import gdown


def safe_extract_filter(tar_info, path):
    """Filter function for safe tar extraction"""
    # Verify there are no absolute paths or parent directory references
    if tar_info.name.startswith(('/', '..')):
        return None
    return tar_info

def download_file(url, local_path):
    """Download a file from URL with progress indication and error handling"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping download.")
        return local_path
        
    try:
        # Extract file ID from the Google Drive URL
        file_id = url.split('id=')[1].split('&')[0]
        output = gdown.download(id=file_id, output=local_path, quiet=False)
        
        if output:
            print(f"\nDownloaded {local_path}")
            return local_path
        else:
            print("Download failed")
            return None
            
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def extract_archive(archive_path, extract_to):
    """Extract tar.gz archive with error handling"""
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_to, filter=safe_extract_filter)
            print(f"Extracted {archive_path} to {extract_to}")
            return True
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    url = "https://drive.usercontent.google.com/download?id=1J73io_KqCoPEAlH3teLWGoZ78yk5n7ll&export=download&authuser=1"  # Replace with your Google Drive URL
    local_archive_path = "data/dataset_papers.tar.gz"
    extract_to = "data"

    # Download and extract
    if download_file(url, local_archive_path):
        if extract_archive(local_archive_path, extract_to):
            print(f"Extraction successful: {os.listdir(extract_to)}")
            os.remove(local_archive_path)
            print(f"Removed {local_archive_path}")
        else:
            print("Extraction failed")
            exit(1)
    else:
        print("Download failed")
        exit(1)