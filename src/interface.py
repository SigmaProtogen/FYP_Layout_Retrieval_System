import panel as pn
import io
from PIL import Image, ImageDraw
from document_analysis import DocumentAnalysis

pn.extension()

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
            pn.Column("## Controls", self.file_input, self.text_input, self.process_button, self.message, width=300),
            self.pdf_pane
        )

        # Document Analysis class

    
    def load_pdf(self, event):
        if self.file_input.value:
            self.pdf_pane.object = io.BytesIO(self.file_input.value)
            self.message.alert_type = "success"
            self.message.object = "PDF loaded successfully."
    
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
        
        # Example function to simulate bounding box retrieval
        bounding_boxes = self.get_bounding_boxes()
        self.update_pdf_with_boxes(bounding_boxes)
    
    def get_bounding_boxes(self):
        # Placeholder function: In reality, you’d get these from a model
        return [(100, 150, 300, 200), (50, 400, 250, 450)]
    
    def update_pdf_with_boxes(self, bounding_boxes):
        pdf_bytes = io.BytesIO(self.file_input.value)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            for bbox in bounding_boxes:
                draw.rectangle(bbox, outline="red", width=3)
            
            # For demonstration, updating only the first page
            break
        
        updated_pdf = io.BytesIO()
        doc.save(updated_pdf)
        self.pdf_pane.object = updated_pdf
        self.message.alert_type = "info"
        self.message.object = "PDF updated with bounding boxes."
    
    def show(self):
        return self.layout

if __name__ == "__main__":
    pdf_app = PDFInterface()
    pdf_app.show().servable()
