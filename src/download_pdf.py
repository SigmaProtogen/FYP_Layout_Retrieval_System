# Functions to download and preprocess PDFs
# Should take in URL, Arxiv ID, or raw PDF upload/filepath
import regex as re
import os
import unstructured


def download_from_url(url):
    return

def download_arxiv(url_or_id, arxiv_id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")):
    # Convert to URL
    if arxiv_id_pattern.match(url_or_id):
        url_or_id = "https://www.arxiv.org/pdf/" + url_or_id + ".pdf"
    return

def download_pdf(url_or_filepath, download_directory="./data/"):
    # If url, preprocess and download from url
    # If filepath, preprocess 
    # Returns processed pdf file, should be standardized regardless of input

    # Check URL/path type
    # Pattern to match a direct arXiv ID or URL
    arxiv_id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
    arxiv_url_pattern = re.compile(r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$")
    if arxiv_id_pattern.match(url_or_filepath) or arxiv_url_pattern.match(url_or_filepath):
        document = download_arxiv(url_or_filepath)
    else:
        # Check for filepath validity
        try:
            if not os.path.exists(url_or_filepath):
                raise FileNotFoundError(f"The file {url_or_filepath} does not exist.")

            with open(url_or_filepath, 'r') as file:
                # Read pdf using unstructured
                process_pdf(filepath)

        except FileNotFoundError as e:
            print(e)

    # Default
    return None    

def process_pdf(filepath):
    return