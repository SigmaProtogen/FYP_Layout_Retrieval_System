from download_pdf import DownloadManager
from pdf_manager import PDFManager
from doi2pdf import doi2pdf

# download_manager =  DownloadManager()
# download_manager.download_pdf(url="https://arxiv.org/abs/1706.03762")
#doi2pdf("10.1109/TAP.2024.3352828", output='data/test.pdf')

pdf_manager = PDFManager(download_directory='./data/')
test_pdf = pdf_manager.read_from_path("./data/1706.03762.pdf")
print('done')