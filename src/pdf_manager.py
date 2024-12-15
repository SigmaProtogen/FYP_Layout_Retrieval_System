import transformers
from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
import pymupdf
from PIL import Image


# Main class for processing
class PDFManager():
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(model_name)
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name, add_prefix_space=True)

        print(f'Using model {self.model}')
    
    # Read and process a PDF file from a path using PyMuPDF
    # If downloaded from arxiv, will call using data dir
    # Else if uploaded, path will be extracted from FileInput or FileSelector
    # Returns a standardized list of images compatible with LayoutLMv3 extractor
    def read_from_path(self, filepath):
        return pymupdf.open(filepath)

    # Create PIL image from PyMuPDF's Pixmap 
    def pixmap_to_pil(self, pixmap):
        # Ensure the pixmap is in RGB mode
        if pixmap.alpha:
            img_data = pixmap.samples
            mode = "RGBA"
        else:
            img_data = pixmap.samples
            mode = "RGB"
        
        return Image.frombytes(mode, (pixmap.width, pixmap.height), img_data)
    
    # Can extract text and bounding boxes using transformers feature extractor
    # Takes in image or batch of images
    # Returns preprocsesed output, text/boxes can be obtained via extractor.words, extractor.boxes
    def extract_features(self, pages):
        page_features = []
        for page in pages:
            page_features.append(self.image_processor(page))
        return page_features
