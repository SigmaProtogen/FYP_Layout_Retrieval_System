import transformers
import torch
from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3ImageProcessor, LayoutLMv3Processor, LayoutLMv3TokenizerFast
import pymupdf
from PIL import Image


# Main class for processing
class PDFManager():
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        print(f'Using model {model_name}')
    
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
    
    # Can extract text and bounding boxes using transformers LayoutLMv3 feature extractor
    # Takes in image or batch of images
    # Returns preprocsesed output, text/boxes can be obtained via extractor.words, extractor.boxes
    def extract_features(self, pages):
        page_features = []
        if not isinstance(pages, list):
            pages = [pages]
        for page in pages:
            page_features.append(self.image_processor(page))
        return page_features


    # UPDATE: Multimodal Chunking of a page
    # Should return a structured element with document and page identifier to link relevant info to source
    # Different modes for ablation study on performance (text, visual, layout, multi)
    def chunk_page(self, mode='multi'):
        # possible mode: Selecting region from LayoutLMv3 first pass and:
            # Extracting any text in the region, Saving the region as an image, Saving layout-dependent objects (e.g. tables)
        
        
        return True


    # Function for visual question answering using LayoutLMv3
    # Uses both text and image modality, extract features and use both features and original image
    def vqa(self, document, question):
        page = document[0] # Initial test: Use first page

        # Extract page image if available
        image = self.pixmap_to_pil(page.get_pixmap(dpi=300))

        # Extract features
        features = self.image_processor(image)
        context = features.words
        boxes = features.boxes

        #question = "Is this a question?"
        # context = ["Example"]
        # boxes = [[0, 0, 1000, 1000]]  # This is an example bounding box covering the whole image.
        document_encoding = self.processor(image, question, context, boxes=boxes, return_tensors="pt")
        outputs = self.model(**document_encoding)

        # Decode answer
        start_idx = torch.argmax(outputs.start_logits, axis=1)
        end_idx = torch.argmax(outputs.end_logits, axis=1)
        answers = self.processor.tokenizer.decode(context[start_idx: end_idx+1]).strip()
        return answers