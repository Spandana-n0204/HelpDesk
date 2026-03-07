import os
import json
from pdf2image import convert_from_path
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


POPPLER_PATH = r"C:\poppler\poppler-23.05.0\Library\bin"

# Folder paths
DOCUMENTS_FOLDER = "../documents"
OUTPUT_FOLDER = "../extracted_data"


def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH
    )

    full_text = ""

    for page in pages:
        text = pytesseract.image_to_string(page)
        full_text += text + "\n"

    # Basic cleaning
    full_text = full_text.replace("\n\n", "\n")
    full_text = full_text.strip()

    return full_text


def process_all_documents():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for file in os.listdir(DOCUMENTS_FOLDER):
        if file.endswith(".pdf"):
            print(f"Processing {file}...")

            pdf_path = os.path.join(DOCUMENTS_FOLDER, file)
            text = extract_text_from_pdf(pdf_path)

            # Split into searchable chunks
            lines = text.split("\n")

            structured_data = []
            for line in lines:
                clean_line = line.strip()
                if len(clean_line) > 25:  # remove junk short lines
                    structured_data.append(clean_line)

            output_path = os.path.join(
                OUTPUT_FOLDER,
                file.replace(".pdf", ".json")
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=4)

    print("\n✅ Documents cleaned and structured successfully.")


if __name__ == "__main__":
    process_all_documents()