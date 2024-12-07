import transformers
from unstructured.partition.pdf import partition_pdf

# Main class for processing
class PDFManager():
    def __init__(self, download_directory):
        self.download_directory = download_directory
    
    # Read and process a PDF file from a path using Unstructured
    # If downloaded from arxiv, will call using data dir
    # Else if uploaded, path will be extracted from FileInput or FileSelector
    def read_from_path(filepath):
        elements = partition_pdf(filepath)

