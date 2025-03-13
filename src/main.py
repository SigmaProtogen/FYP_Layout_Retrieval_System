from document_analysis import DocumentAnalysis
from pdf_manager import PDFManager
# from doi2pdf import doi2pdf

import panel as pn
from interface import PDFViewerInterface

if __name__ == "__main__":
    # Activate Panel extensions
    pn.extension(sizing_mode="stretch_width")

    # Create an instance of the PDF viewer interface
    viewer = PDFViewerInterface()

    # Serve the application
    viewer.get_layout().servable("PDF Viewer")


# Sample run for 1 document
# For debug, run pipeline.faiss_read() in cell below to prevent rereading doc
pipeline = DocumentAnalysis()
doc_path = "../data/1706.03762.pdf"
doc = pipeline.read_from_path(doc_path)

# Processing for each page
# Remove [:3] for entire doc, keep for testing
for page_idx, page in enumerate(doc):
    blocks = pipeline.detect_layout(page)

    # Processing for each block to be vectorized
    for b in blocks:
        # Process as an image if detected type is a figure, else process as text
        if b.type == "Figure":
            # Crop and get image embeddings
            segmented_image = pipeline._crop_image(page, b, padding=20)
            image_embs = pipeline.encode_image(segmented_image)
            pipeline.add_to_faiss(embedding=image_embs, content="Figure", content_type=b.type, page_idx=page_idx, bbox=b.block.coordinates)
        else:
            # Chunk text and create new blocks, and process for each block
            # Returns list even if not chunked
            chunks = pipeline.chunk_text(b.text)

            for chunk in chunks:
                b2 = deepcopy(b)
                b2.set(text=chunk, inplace=True)

                # Create duplicate blocks for each chunk
                # Encode using text and add to FAISS
                text_embs = pipeline.encode_text(b2.text)
                pipeline.add_to_faiss(embedding=text_embs, content=b2.text, content_type=b.type, page_idx=page_idx, bbox=b2.block.coordinates)
            