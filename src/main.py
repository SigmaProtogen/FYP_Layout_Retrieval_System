from download_pdf import DownloadManager
from pdf_manager import PDFManager
from doi2pdf import doi2pdf
import panel as pn
from interface import PDFViewerInterface

if __name__ == "__main__":
    # Activate Panel extensions
    pn.extension(sizing_mode="stretch_width")

    # Create an instance of the PDF viewer interface
    viewer = PDFViewerInterface()

    # Serve the application
    viewer.get_layout().servable("PDF Viewer")

pdf_manager = PDFManager(model_name = "rubentito/layoutlmv3-base-mpdocvqa")
# pdf_manager = PDFManager(model_name = "microsoft/layoutlmv3-base")
test_pdf = pdf_manager.read_from_path("./data/1706.03762.pdf")

# print(test_pdf.page_count)
# pages = [page.get_pixmap(dpi=300) for page in test_pdf]
# pil_pages = [pdf_manager.pixmap_to_pil(page) for page in pages]
# features = pdf_manager.extract_features(pil_pages)
# print(features)

question = "What is the title of the paper?"
ans = pdf_manager.vqa(test_pdf, question)
print(ans)





