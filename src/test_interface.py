import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import io
import pymupdf
from document_analysis import DocumentAnalysis

pn.extension(template='fast')
document_processor = DocumentAnalysis()

# Functions to load/process/update here
def load_pdf(event):
    print('uploaded')
    if file_input.value:
        message.alert_type = "info"
        message.object = "Processing PDF..."

        # Process PDF using processor
        # Get list of pages as images
        pdf_bytes = io.BytesIO(file_input.value)
        pdf_pane.object = pdf_bytes

        #pages = document_processor.read_from_bytes(pdf_bytes.getvalue())  # Load PDF pages
        document_processor.faiss_read() # Test read
        
        message.alert_type = "success"
        message.object = "PDF loaded successfully."

def process_query(event):
    if not file_input.value:
        message.alert_type = "danger"
        message.object = "Please upload a PDF first."
        return
        
    query = text_input.value.strip()
    if not query:
        message.alert_type = "warning"
        message.object = "Please enter a query."
        return
    
    message.alert_type = "info"
    message.object = "Processing query..."
    # Process document and retrieve bounding boxes
    # Answer is in list[dict] format for bbox and fulltext extraction
    answers = document_processor.search_faiss(query, n=3)
    print(answers)
    
    # Update pdf with bounding boxes from answer and annotate on 
    bboxes = [a['bbox'] for a in answers]
    update_pdf_with_boxes(bboxes)
    
def update_pdf_with_boxes(bboxes):
    pdf_bytes = io.BytesIO(file_input.value)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    
    for page_idx, page in enumerate(doc):
        for bbox in bounding_boxes:
            if bbox["page"] == page_idx:  # Draw only on the correct page
                rect = pymupdf.Rect(bbox["bbox"])
                page.insert_rect(rect, color=(1, 0, 0), width=3)
        
    updated_pdf = io.BytesIO()
    # doc.save(updated_pdf)
    updated_pdf.seek(0)
    
    pdf_pane.object = updated_pdf
    message.alert_type = "info"
    message.object = "PDF updated with bounding boxes."


# PDF Upload on sidebar (FileInput)
file_input = pn.widgets.FileInput(
    accept='.pdf', name='Select a PDF', multiple=False
).servable(target='sidebar')

# Query input on sidebar
text_input = pn.widgets.TextInput(
    name='Enter Query', placeholder='Type your query here...'
).servable(target='sidebar')

process_button = pn.widgets.Button(
    name='Process', button_type='primary'
).servable(target='sidebar')

# Message alert on sidebar too
message = pn.pane.Alert(
    "Upload a PDF to start.", alert_type="info"
).servable(target='sidebar')


# PDF Display on main
# Placeholder, to be changed with load_pdf
pdf_pane = pn.pane.PDF(
    'https://arxiv.org/pdf/1706.03762', width=1100, height=900, embed=True
)
pn.Column(pdf_pane).servable(target='main')

# Bind functions
pn.bind(load_pdf, file_input, watch=True)
process_button.on_click(process_query)