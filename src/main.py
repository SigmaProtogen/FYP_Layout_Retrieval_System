from document_analysis import DocumentAnalysis
# from pdf_manager import PDFManager
import panel as pn
from interface import PDFInterface

# Initialize the PDF interface
pdf_app = PDFInterface()

# Serve the application
pn.serve(pdf_app.show(), title="PDF Annotator", show=True)





# Document analysis to be added as part of interface.py actions
# # Sample run for 1 document
# pipeline = DocumentAnalysis()
# doc_path = "./data/1706.03762.pdf"
# doc = pipeline.read_from_path(doc_path)
# pipeline.process_document(doc)

# # Retrieval given query
# # Query should be obtained from panel user text input
# query="How does a Transformer use positional encoding?"
# answer = pipeline.search_faiss(query)
# print(answer)
# pipeline.faiss_persist()