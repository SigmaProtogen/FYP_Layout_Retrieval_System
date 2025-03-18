import panel as pn
import pymupdf  # PyMuPDF
import io
from document_analysis import DocumentAnalysis  # Import the DocumentAnalysis class

pn.extension()

document_processor = DocumentAnalysis()

class PDFInterface:
    def __init__(self):
        self.file_input = pn.widgets.FileInput(accept='.pdf', name='Upload PDF')
        self.text_input = pn.widgets.TextInput(name='Enter Query', placeholder='Type your query here...')
        self.process_button = pn.widgets.Button(name='Process', button_type='primary')
        
        self.pdf_pane = pn.pane.PDF(None, width=600, height=800)
        self.message = pn.pane.Alert("Upload a PDF to start.", alert_type="info")
        
        self.file_input.param.watch(self.load_pdf, 'value')
        self.process_button.on_click(self.process_query)
        
        self.layout = pn.Row(
            pn.Column("## Control", self.file_input, self.text_input, self.process_button, self.message, width=300),
            self.pdf_pane
        )
    
    def load_pdf(self, event):
        if self.file_input.value:
            pdf_bytes = io.BytesIO(self.file_input.value)
            self.pages = document_processor.read_from_bytes(pdf_bytes.getvalue())  # Load PDF pages
            self.pdf_pane.object = pdf_bytes
            self.message.alert_type = "success"
            self.message.object = "PDF loaded successfully."
            self.layout.show()
    
    def process_query(self, event):
        if not self.file_input.value:
            self.message.alert_type = "danger"
            self.message.object = "Please upload a PDF first."
            return
        
        query = self.text_input.value.strip()
        if not query:
            self.message.alert_type = "warning"
            self.message.object = "Please enter a query."
            return
        
        # Process document and retrieve bounding boxes
        document_processor.process_document(self.pages)
        results = document_processor.search_faiss(query)
        bounding_boxes = [result['bbox'] for result in results]
        
        self.update_pdf_with_boxes(bounding_boxes)
    
    def update_pdf_with_boxes(self, bounding_boxes):
        pdf_bytes = io.BytesIO(self.file_input.value)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        for page_idx, page in enumerate(doc):
            for bbox in bounding_boxes:
                if bbox["page"] == page_idx:  # Draw only on the correct page
                    rect = pymupdf.Rect(bbox["bbox"])
                    page.insert_rect(rect, color=(1, 0, 0), width=3)
            
        updated_pdf = io.BytesIO()
        doc.save(updated_pdf)
        updated_pdf.seek(0)
        
        self.pdf_pane.object = updated_pdf
        self.message.alert_type = "info"
        self.message.object = "PDF updated with bounding boxes."
    
    def show(self):
        return self.layout

