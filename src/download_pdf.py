# Functions to download and process PDFs
# Should take in URL, Arxiv ID, or raw PDF upload/filepath
import regex as re
import os
import requests
from doi2pdf import doi2pdf

class DownloadManager():
    def __init__(self, download_directory="./data/"):
        """
        Initialize download manager with specified download directory.
        Parameters:
        - download_directory: Directory where files will be downloaded to.
        """
        self.download_directory = download_directory
        os.makedirs(self.download_directory, exist_ok=True)

    def download_pdf(self, url, filename=None):
        """
        Download a file from the given URL and save it to the specified directory.
        Parameters:
        - url: URL of the file to be downloaded.
        - filename: Name to save the file as. If None, it uses the name from the URL.

        Returns:
        Full filepath to the downloaded file or an error message.
        """
        arxiv_id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
        arxiv_url_pattern = re.compile(r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$")
        doi_url_pattern = re.compile(r"")

        # URL Preprocessing for specific formats
        # Should convert id to link, abs to pdf for arxiv
        if arxiv_id_pattern.match(url):
            # Convert id into pdf link
            url = "https://www.arxiv.org/pdf/" + url + ".pdf"
        elif arxiv_url_pattern.match(url) and 'abs' in url:
            # Convert abstract into pdf for direct download
            url_segments = url.split('abs')
            url_segments.insert(1, 'pdf')
            url = ''.join(url_segments)

        # If a DOI link is identified, use doi2pdf to download instead
        # Else, use regular requests
        try:
            # Send a GET request to the URL
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP errors
            
            # Determine the file name
            if filename is None:
                filename = url.split("/")[-1]
            if filename[-4:] != '.pdf': filename += '.pdf'
            
            # Full path to save the file
            file_path = os.path.join(self.download_directory, filename)
            
            # Write the file content to the specified path
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            return f"File downloaded successfully: {file_path}"
        
        except requests.exceptions.RequestException as e:
            return f"Error downloading file: {e}"