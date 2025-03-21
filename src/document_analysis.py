import layoutparser as lp
import pymupdf
from PIL import Image
import cv2
import numpy as np
import json
from tqdm import tqdm

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
import voyageai

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["VOYAGE_API_KEY"] = "pa-t-QdSeBOYxYQ83TObGLxkR4iqMZpYylSWOLBmthFUG7"

device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Unified class for processing, analyzing and storing a document
class DocumentAnalysis():
    def __init__(self, embedding_model = "voyageai/voyage-multimodal-3", cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2", vector_dir = "./data/.vectorstore/"):
         # Layout detection
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                                 device=device)
        self.ocr_agent = lp.TesseractAgent(languages='eng') 

        # Dual encoders for embeddings
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.text_splitter = NLTKTextSplitter.from_huggingface_tokenizer(self.embedding_tokenizer, chunk_size=256, chunk_overlap = 10)
        self.vo_embeddings = voyageai.Client()

        # Cross encoder for retrieval-reranking
        self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model).to(device)       

        # Vectorstore variables
        self.dimension = 1024  # Voyage's embedding size
        self.faiss_index = faiss.IndexFlatL2(self.dimension) # FAISS Vector store
        self.metadata_store = {}  # Store mapping of IDs and document page number to content
        self.vector_dir = vector_dir # Directory to write data to

    # Read a PDF document using PyMuPDF
    # Returns list of page images in cv2 format
    def read_from_path(self, filepath):
        doc = pymupdf.open(filepath)
        return [self.pixmap_to_cv2(page.get_pixmap(dpi=200)) for page in doc]

    # Function to read pdf from bytestream
    # To be used with Panel's FileInput
    def read_from_bytes(self, pdf_bytes):
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return [self.pixmap_to_cv2(page.get_pixmap(dpi=250)) for page in doc]

    # Convert PyMuPDF pixmap to cv2
    def pixmap_to_cv2(self, pixmap):
        bytes = np.frombuffer(pixmap.samples, dtype=np.uint8)
        image = bytes.reshape(pixmap.height, pixmap.width, pixmap.n)
        image = image[..., ::-1]
        return image

    # Convert PIL jpeg to cv2
    def pil_to_cv2(self, pil_image):
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy() # RBG to BGR
        return open_cv_image


    # Takes in image object from read_from_path()
    # Detects layout -> Processes ROI by label
    def detect_layout(self, image):
        layout = self.model.detect(image)

        # Separate boxes by category
        text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
        title_blocks = lp.Layout([b for b in layout if b.type=='Title'])
        list_blocks = lp.Layout([b for b in layout if b.type=='List'])
        table_blocks = lp.Layout([b for b in layout if b.type=='Table'])
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

        # Processing text blocks
        # Sourced from LayoutParser's Deep Layout Analysis example
        # Eliminate text blocks nested in images/figures
        text_blocks = lp.Layout([b for b in text_blocks \
                        if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
        # Sort boxes
        h, w = image.shape[:2]
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
        # The b.coordinates[1] corresponds to the y coordinate of the region
        # sort based on that can simulate the top-to-bottom reading order 
        right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

        # And finally combine the two lists and add the index
        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

        # Perform OCR to extract text
        for block in text_blocks + title_blocks + list_blocks + table_blocks + figure_blocks:
            # Add padding in each image segment to improve robustness
            text = self._ocr_on_block(image, block)
            block.set(text=text, inplace=True) # Assign parsed text to block element
            
        # Return all blocks on the page as a list
        return text_blocks + title_blocks + list_blocks + table_blocks + figure_blocks

    # Function to crop an image given block's bbox and additional padding
    def _crop_image(self, image, block, padding=10):
        return (block.pad(left=padding, right=padding, top=padding, bottom=padding).crop_image(image))

    # Perform OCR to extract text given image and block (for text, tables and lists)
    def _ocr_on_block(self, image, block):
        # Add padding in each image segment to improve robustness
        segment_image = (block.pad(left=5, right=5, top=5, bottom=5).crop_image(image))
        return self.ocr_agent.detect(segment_image)

    # Vectorstore functions
    # Function to chunk text to set length
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks

    # Test voyage embeddings
    def get_voyage_embeddings(self, text, image=None):
        # Convert image from array into PIL
        if image is not None: image = Image.fromarray(np.uint8(image))
        if (text is None or len(text)==0) and image is not None:
            inputs = [[image]]
        elif image is None:
            inputs = [[text]]
        else:
            inputs = [[text, image]]
        result = self.vo_embeddings.multimodal_embed(inputs, model="voyage-multimodal-3")
        return np.array(result.embeddings)

    # Function to add item to FAISS
    # Specify content, type, page and bounding box from blocks
    def add_to_faiss(self, embedding, content, chunk, content_type, page_idx, bbox):
        idx = len(self.metadata_store)  # Assign unique index
        self.faiss_index.add(embedding)
        self.metadata_store[idx] = {"type": content_type, "content": content, "chunk": chunk, "page": page_idx, "bbox": bbox}
    
    # Perform retrieval and reranking
    def search_faiss(self, query, k=20, n=5):
        query_embedding = self.get_voyage_embeddings(query)
        _, indices = self.faiss_index.search(query_embedding, k)
        indices = [int(i) for i in indices[0]]
        
        # Cross encoder reranking on chunk contents
        answers = [self.metadata_store[idx] for idx in indices]
        answer_chunks = [a['content'] for a in answers]
        queries = [query for i in range(len(answers))] # Repeat for tokenizer input
        features = self.cross_encoder_tokenizer(queries, answer_chunks,  padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): # Rerank
            scores = self.cross_encoder(**features).logits

        # Get indices of the top n scores
        best_indices = np.argsort(np.array(scores.flatten()))[-n:][::-1]  # Sort and reverse

        # Retrieve responses using the indices
        best_answers = [answers[i] for i in best_indices]
        return best_answers


    # Writes the vectorstore and metadata into a given path
    def faiss_persist(self, subdir = ''):
        full_dir = self.vector_dir + subdir
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        faiss.write_index(self.faiss_index, full_dir+"faiss.index")
        json.dump(self.metadata_store, open(full_dir+"metadata.json", 'w'))
    
    # Read from existing vector stores
    def faiss_read(self, subdir = ''):
        full_dir = self.vector_dir + subdir
        if not os.path.exists(full_dir):
            print("Directory does not exist")
            return False
        self.faiss_index = faiss.read_index(full_dir+"faiss.index")
        self.metadata_store = json.load(open(full_dir+"metadata.json", 'r'), object_hook=self._convert_keys)
    
    # Convert keys from string to int when deserializing
    def _convert_keys(self, d):
        return {int(k) if k.isdigit() else k: v for k, v in d.items()}

    # Function to process all pages of a document given all the functions above
    # Returns nothing, processes and ingests document into the object's metadata store
    def process_document(self, doc):
        for page_idx, page in enumerate(tqdm(doc)):
            blocks = self.detect_layout(page)

            # Processing for each block to be vectorized
            for b in blocks:
                chunks = self.chunk_text(b.text) if b.text is not None else None
                for chunk in chunks:
                    if b.type in ['Figure', 'Table']:
                        segmented_image = self._crop_image(page, b, padding=20)
                        embs = self.get_voyage_embeddings(chunk, segmented_image)
                    else:
                        embs = self.get_voyage_embeddings(chunk)
                    self.add_to_faiss(
                        embedding=embs, 
                        content=b.text, 
                        chunk=chunk, 
                        content_type=b.type, 
                        page_idx=page_idx, 
                        bbox=b.block.coordinates
                    )