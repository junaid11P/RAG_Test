import re
import os
from typing import List

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

class DocumentProcessor:
    """
    Unified Document Processor using Microsoft MarkItDown.
    Handles PDF, DOCX, DOC, XLSX, PPTX, CSV, HTML, Images, etc.
    """
    _md = MarkItDown() if MarkItDown else None

    @staticmethod
    def basic_clean(text: str) -> str:
        """Cleans text while preserving structural characters like | for tables."""
        # Keep ASCII characters + some table markers
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        # Normalize horizontal spaces but keep structure
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def remove_noise(text: str) -> str:
        """Removes common digital noise without destroying table structure."""
        noise_patterns = [
            r"mailto:.*",
            r"https?://\S+",
            r"www\.\S+",
            r"copyright\s+©\s+\d{4}",
            r"page\s+\d+\s+of\s+\d+",
            r"(?i)confidential",
            r"-----------------+"
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    def process_document(cls, file_path: str) -> str:
        """
        Complete production pipeline using MarkItDown.
        """
        if not cls._md:
            return "[Error: MarkItDown library not installed or initialized.]"

        try:
            # MarkItDown handles almost all formats automatically
            result = cls._md.convert(file_path)
            if not result or not result.text_content:
                return "No readable text found in document."

            # Clean the markdown content
            clean_text = cls.basic_clean(result.text_content)
            final_text = cls.remove_noise(clean_text)
            
            if not final_text.strip():
                return "Document appeared empty after processing."
                
            return final_text

        except Exception as e:
            print(f"MarkItDown conversion error: {e}")
            return f"[Processing Error: {str(e)}]"
