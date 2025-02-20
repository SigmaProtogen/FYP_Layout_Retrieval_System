import panel as pn
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO


class PDFViewerInterface:
    def __init__(self):
        # Initialize widgets
        self.pdf_path_input = pn.widgets.FileInput(name="Upload PDF", accept=".pdf")
        self.page_slider = pn.widgets.IntSlider(name="Page", start=0, end=0, value=0)

        # Set up layout
        self.sidebar = pn.Column(
            "## Sidebar",
            self.pdf_path_input,
            self.page_slider,
            width=250,
        )
        self.main_content = pn.Column(
            "## PDF Viewer",
            self.render_pdf,
            width=700,
        )
        self.layout = pn.Row(self.sidebar, self.main_content)

        # Attach widget dependencies
        self.pdf_path_input.param.watch(self.update_page_slider, "value")
        self.page_slider.param.watch(self.render_pdf, "value")

        # Internal state
        self.temp_pdf_path = "/tmp/temp.pdf"

    def extract_pdf_pages(self, pdf_path):
        # Replace with pdf_manager implementation
        """
        Extracts all pages of a PDF as images.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list: List of PIL images, one for each page.
        """
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pixmap = page.get_pixmap(dpi=150)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            pages.append(image)

        doc.close()
        return pages

    def update_page_slider(self, event):
        """
        Updates the page slider's range based on the uploaded PDF.
        """
        if event.new:
            # Save the uploaded PDF to a temporary file
            with open(self.temp_pdf_path, "wb") as f:
                f.write(event.new)

            # Update slider range based on the number of pages
            doc = fitz.open(self.temp_pdf_path)
            num_pages = len(doc)
            doc.close()

            self.page_slider.start = 0
            self.page_slider.end = num_pages - 1
            self.page_slider.value = 0

    def render_pdf(self, event=None):
        """
        Renders the currently selected PDF page as an image in Panel.
        """
        if self.pdf_path_input.value:
            # Extract pages from the temporary PDF file
            pages = self.extract_pdf_pages(self.temp_pdf_path)
            page_index = self.page_slider.value
            img = pages[page_index]

            # Display the image
            self.main_content[:] = [pn.pane.PNG(img, width=700, height=900)]
        else:
            self.main_content[:] = [pn.pane.Markdown("**Upload a PDF to display its pages.**")]

    def get_layout(self):
        return self.layout
