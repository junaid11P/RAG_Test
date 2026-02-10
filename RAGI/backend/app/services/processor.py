import re
import fitz  # PyMuPDF
import docx
import os

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extracts text from a DOCX file."""
        doc = docx.Document(docx_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)

    @staticmethod
    def extract_text_from_txt(txt_path: str) -> str:
        """Extracts text from a TXT file."""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def basic_clean(text: str) -> str:
        """Basic text cleaning: ASCII only and whitespace normalization."""
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def remove_noise(text: str) -> str:
        """Removes common noise patterns from resume/documents."""
        noise_patterns = [
            r"mailto:.*",
            r"https?://\S+",
            r"www\.\S+",
            r"copyright\s+©\s+\d{4}",
            r"all rights reserved",
            r"page\s+\d+\s+of\s+\d+",
            r"(?i)confidential",
            r"-----------------+"
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    def process_document(cls, file_path: str) -> str:
        """Complete pipeline: extract based on ext -> basic clean -> remove noise."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            raw_text = cls.extract_text_from_pdf(file_path)
        elif ext == '.docx' or ext == '.doc':
            # Note: .doc usually needs more complex handling, but for MVP we'll treat as docx
            # Better to use a library that handles both or tell user docx is preferred
            raw_text = cls.extract_text_from_docx(file_path)
        elif ext == '.txt':
            raw_text = cls.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        clean_text = cls.basic_clean(raw_text)
        final_text = cls.remove_noise(clean_text)
        return final_text
