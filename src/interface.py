import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import io
import pymupdf
import copy
from document_analysis import DocumentAnalysis

pn.extension(template='fast')
document_processor = DocumentAnalysis(vector_dir="./data/.vectorstore/")

# Functions to load/process/update here
# Function to load PDF uploaded from fileInput
def load_pdf(event):
    if file_input.value:
        filename = file_input.filename

        message.alert_type = "info"
        message.object = "Processing PDF..."

        # Process PDF using processor
        # Get list of pages as images
        pdf_bytes = io.BytesIO(file_input.value)
        pdf_pane.object = pdf_bytes
        pdf_pane.param.trigger("object")

        # Tries to read from the file's vectorstore
        # If it doesn't exist, process then persist
        if not document_processor.faiss_read(subdir=filename):
            # Process pages and ingest
            message.object = "Processing new document..."
            doc = document_processor.read_from_bytes(pdf_bytes.getvalue())
            document_processor.process_document(doc)
            document_processor.faiss_persist(subdir=filename)
        
        message.alert_type = "success"
        message.object = "PDF loaded successfully."

def process_query(event):
    global pdf_bytes
    if file_input.value:
        pdf_bytes = io.BytesIO(file_input.value)  # Update pdf_bytes if a new file is uploaded
    elif pdf_bytes is None:  # No uploaded file and no existing PDF
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
    # Uses chunk_slider value to specify amount of chunks
    # Answer is in list[dict] format for bbox and fulltext extraction
    answers = document_processor.search_faiss(query, n=chunk_slider.value)
    
    # Update pdf with bounding boxes from answer and annotate on 
    # Pass answers in to use additional info
    update_pdf_with_boxes(answers)
    
def update_pdf_with_boxes(answers):
    # Load document object from bytes
    pdf_bytes.seek(0)
    doc = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    
    for i, ans in enumerate(answers):
        bbox_scaled = tuple([(9/25)*coord for coord in ans['bbox']])

        rect = pymupdf.Rect(bbox_scaled)
        # Color fades from (1,0,0) to (1,0.5,0.5) in order of answer relevance
        gb = 0 if i==0 else 0.5
        doc[ans["page"]].draw_rect(rect, color=(1, gb, gb), width=3)

    # Save the document into a bytestream for PDF pane
    updated_pdf = io.BytesIO()
    doc.save(updated_pdf)
    updated_pdf.seek(0)
    pdf_pane.object = updated_pdf
    pdf_pane.start_page = answers[0]["page"]+1 # Immediately go to page with most relevant bbox
    
    message_text = f"PDF updated with bounding boxes.\n\nTop retrieved chunk:\n\"{answers[0]['chunk']}\""
    message.alert_type = "info"
    message.object = message_text


# Sidebar components
file_input = pn.widgets.FileInput(name='Upload a PDF', accept='.pdf', multiple=False, align='center')
text_input = pn.widgets.TextInput(name='Enter Query', placeholder='Type your query here...')
chunk_slider = pn.widgets.IntSlider(name="Chunks to return", start=1, end=10, step=1, value=5)
process_button = pn.widgets.Button(name='Process query', button_type='primary', align='center').servable()
message = pn.pane.Alert("Upload a PDF or Enter a query to start.", alert_type="info")



# PDF Display on main
# Default PDF to display on startup
default_pdf_name = "1706.03762.pdf"
default_pdf_path = "./data/"+default_pdf_name
pdf_pane = pn.pane.PDF(
    default_pdf_path, width=1100, height=900, embed=True
)

# Bind functions
pn.bind(load_pdf, file_input, watch=True)
process_button.on_click(process_query)

# Loading default document
document_processor.read_from_path(default_pdf_path) # Load default document as class object
document_processor.faiss_read(default_pdf_name)
pdf_bytes = io.BytesIO()
document_processor.document.save(pdf_bytes)
pdf_bytes.seek(0)

interface_layout = pn.template.MaterialTemplate(
    title="Document Analysis",
    main=[pn.Column(pdf_pane)],
    sidebar=[pn.Column(file_input, text_input, chunk_slider, process_button, message).servable()]
).servable()

# Utility to stop session and cleanup
def stop_server(session_context):
    print("Session terminated, shutting down server")
    server.stop()

def serve_application():
    pn.state.on_session_destroyed(lambda session_context: stop_server(server))
    interface_layout.servable()
    # Serve application directly
    # Create as a server object to be stopped
    global server
    server = pn.serve(
        interface_layout, 
        port=5006, threaded=True, title="Document Analysis", show=True,
        theme_toggle=False
    )
