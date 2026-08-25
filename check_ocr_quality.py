"""
Quick OCR sanity check — run this on ONE known-scanned PDF before trusting
the full indexing pipeline. It shows you exactly what Tesseract reads off
a given page so you can judge quality with your own eyes.

Usage:
    python check_ocr_quality.py path/to/file.pdf 3
    (checks page 3, 1-indexed)
"""
import sys
import io
import fitz
import pytesseract
from PIL import Image

def check_page(pdf_path, page_num_1indexed, lang="amh+eng", dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[page_num_1indexed - 1]

    raw_text = page.get_text().strip()
    print(f"--- Embedded text layer ({len(raw_text)} chars) ---")
    print(raw_text[:300] if raw_text else "(none — page is likely scanned)")

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img.save("ocr_check_page.png")  # save so you can look at the actual scan
    print(f"\n(saved rendered page image to ocr_check_page.png for visual comparison)")

    ocr_text = pytesseract.image_to_string(img, lang=lang).strip()
    print(f"\n--- OCR output ({len(ocr_text)} chars, lang={lang}) ---")
    print(ocr_text[:1000])

    doc.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_ocr_quality.py <pdf_path> <page_number>")
        sys.exit(1)
    check_page(sys.argv[1], int(sys.argv[2]))
