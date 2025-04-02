from util import docling
import glob

input_path = "./assets/library"
out_path = "./assets/library/docling_out"

input_pdfs = glob.glob(f"{input_path}/*.pdf")

if __name__ == "__main__":
    converter = docling.PDFConverter()
    converter.convert_pdf(input_pdfs, out_path)
