# Unified class
class DocumentAnalysis():
    def __init__(self, embedding_model = "openai/clip-vit-base-patch32", cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        # Layout detection
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.ocr_agent = lp.TesseractAgent(languages='eng') 

        # Dual encoders for embeddings
        self.clip_model = CLIPModel.from_pretrained(embedding_model)
        self.clip_processor = CLIPProcessor.from_pretrained(embedding_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, chunk_size=77, chunk_overlap=5)

        # Cross encoder for retrieval-reranking
        self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)        

        # Vectorstore variables
        self.dimension = 512  # CLIP's embedding size
        self.faiss_index = faiss.IndexFlatL2(self.dimension) # FAISS Vector store
        self.metadata_store = {}  # Store mapping of IDs and document page number to content
        self.vector_dir = '../data/.vectorstore/' # Directory to write data to

    # Read a PDF document using PyMuPDF
    # Returns list of page images in cv2 format
    def read_from_path(self, filepath):
        doc = pymupdf.open(filepath)
        return [self.pixmap_to_cv2(page.get_pixmap(dpi=300)) for page in doc]

    # Convert PyMuPDF pixmap to cv2
    def pixmap_to_cv2(self, pixmap):
        bytes = np.frombuffer(pixmap.samples, dtype=np.uint8)
        image = bytes.reshape(pixmap.height, pixmap.width, pixmap.n)
        image = image[..., ::-1]
        return image

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
        for block in text_blocks + title_blocks + list_blocks + table_blocks:
            # Add padding in each image segment to improve robustness
            text = self._ocr_on_block(image, block)
            block.set(text=text, inplace=True) # Assign parsed text to block element
            
        # Return all blocks on the page as a list
        # Omit titles as it affects retrieval
        return text_blocks + list_blocks + table_blocks + figure_blocks

    # Function to crop an image given block's bbox and additional padding
    def _crop_image(self, image, block, padding=10):
        return (block.pad(left=padding, right=padding, top=padding, bottom=padding).crop_image(image))

    # Perform OCR to extract text given image and block (for text, tables and lists)
    def _ocr_on_block(self, image, block):
        # Add padding in each image segment to improve robustness
        segment_image = (block.pad(left=5, right=5, top=5, bottom=5).crop_image(image))
        return self.ocr_agent.detect(segment_image)

    # Vectorstore functions
    # Function to chunk text to CLIP max length
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks

    # Function to encode text
    def encode_text(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs).numpy()
        return embedding / np.linalg.norm(embedding)  # Normalize

    # Function to encode image
    def encode_image(self, image):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs).numpy()
        return embedding / np.linalg.norm(embedding)  # Normalize

    # Function to add item to FAISS
    # Specify content, type, page and bounding box from blocks
    def add_to_faiss(self, embedding, content, content_type, page_idx, bbox):
        idx = len(self.metadata_store)  # Assign unique index
        self.faiss_index.add(embedding)
        self.metadata_store[idx] = {"type": content_type, "content": content, "page": page_idx, "bbox": bbox}
    
    # Perform retrieval (and rerank)
    def search_faiss(self, query, k=10):
        query_embedding = self.encode_text(query)
        _, indices = self.faiss_index.search(query_embedding, k)
        # Convert to int (faiss_read may change it to numpy)
        indices = [int(i) for i in indices[0]]
        
        # Display retrieved items
        # retrieved items accessed by metadata_store using fetched indices
        for idx in indices:
            print(f"Retrieved {self.metadata_store[idx]['type']}: {self.metadata_store[idx]['content']}")
        
        # Cross encoder reranking on text modality
        answers = [self.metadata_store[idx] for idx in indices if self.metadata_store[idx]['type']!='Figure']
        answer_texts = [a['content'] for a in answers]
        queries = [query for i in range(len(answers))] # Repeat for tokenizer input
        features = self.cross_encoder_tokenizer(queries, answer_texts,  padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): # Rerank
            scores = self.cross_encoder(**features).logits

        # Select index with best score
        best_index = np.argmax(scores)
        best_answer = answers[best_index] # Answer with full metadata for downstream

        # Display for debug
        print(scores, best_answer)


    # Writes the vectorstore and metadata into a given path
    def faiss_persist(self):
        faiss.write_index(self.faiss_index, self.vector_dir+"faiss.index")
        json.dump(self.metadata_store, open(self.vector_dir+"metadata.json", 'w'))
    
    # Read from existing vector stores
    def faiss_read(self):
        self.faiss_index = faiss.read_index(self.vector_dir+"faiss.index")
        self.metadata_store = json.load(open(self.vector_dir+"metadata.json", 'r'), object_hook=self._convert_keys)
    
    # Convert keys from string to int when deserializing
    def _convert_keys(self, d):
        return {int(k) if k.isdigit() else k: v for k, v in d.items()}