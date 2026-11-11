
from pathlib import Path
from pypdf import PdfReader

PDF_FOLDER = Path("data/policies")

for pdf_file in sorted(PDF_FOLDER.glob("*.pdf")):
    print("\n" + "=" * 70)
    print(f"FILE: {pdf_file.name}")
    print("=" * 70)

    reader = PdfReader(str(pdf_file))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        print(f"\n--- PAGE {page_number} ---")
        print(text)
