import transformers
from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3ImageProcessor, LayoutLMv3Processor, LayoutLMv3TokenizerFast
import pymupdf
from PIL import Image


# Main class for processing
class PDFManager():
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.model = LayoutLMv3ForQuestionAnswering.from_pretrained("HYPJUDY/layoutlmv3-base-finetuned-publaynet")
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base", add_prefix_space=True)

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

    # Function for visual question answering using LayoutLMv3
    # Uses both text and image modality, extract features and use both features and original image
    def vqa(self, document, query):
        page = document[0] # Initial test: Use first page

        words = page.get_text("words")  # Returns a list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
        text_data = []
        bboxes = []

        
        for word in words:
            x0, y0, x1, y1, text, *_ = word
            text_data.append(text)
            bboxes.append([int(x0), int(y0), int(x1), int(y1)])
        
        # Extract page image if available
        image = self.pixmap_to_pil(page.get_pixmap(dpi=300))

        #features = self.extract_features(page)
        words = page.get_text("words")  # Returns a list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
        text_data = []
        bboxes = []
        for word in words:
            x0, y0, x1, y1, text, *_ = word
            text_data.append(text)
            bboxes.append([int(x0), int(y0), int(x1), int(y1)])

        inputs = self.processor(
            text=query,
            #boxes=bboxes,
            images=image,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        # Process query into inputs
        #inputs.update(self.processor(text=query, return_tensors="pt"))

        # Inference and decoding
        outputs = self.model(**inputs)
        answer_start = outputs.start_logits.argmax(dim=-1)
        answer_end = outputs.end_logits.argmax(dim=-1)

        ans = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
        return ans