import re
import os
import io
import uuid
from typing import List, Optional
from app.db.mongodb import db

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

class DocumentProcessor:
    @classmethod
    def get_md_engine(cls):
        """Initializes MarkItDown without vision capabilities to ignore images."""
        if not MarkItDown:
            return None
            
        return MarkItDown()

    @staticmethod
    def basic_clean(text: str) -> str:
        """Cleans text while preserving structural characters like | for tables."""
        # Remove common icon/image placeholders that might be left as text
        text = re.sub(r"\[image\]|\[icon\]|\[pic\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
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
    async def process_document(cls, file_path: str, user_id: str, doc_id: str) -> str:
        """
        Complete production pipeline using MarkItDown.
        Ignores images and icons while preserving text and tables.
        Runs in a separate thread to prevent blocking the event loop.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        md = cls.get_md_engine()
        if not md:
            return "[Error: MarkItDown library not installed.]"

        try:
            print(f"DEBUG: Starting conversion for {file_path}...")
            # 1. Convert document to Markdown (Run in thread to prevent blocking)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, md.convert, file_path)
            
            if not result or not result.text_content:
                return "No readable text found in document."

            text_content = result.text_content
            print(f"DEBUG: Conversion complete. Content length: {len(text_content)}")

            # 2. Remove all image/icon references (Markdown image tags)
            # This ensures images are ignored in the final storage
            # We use a more robust regex for markdown images
            text_content = re.sub(r"!\[.*?\]\(.*?\)", "", text_content)
            
            # 3. Clean and Finalize
            clean_text = cls.basic_clean(text_content)
            final_text = cls.remove_noise(clean_text)
            
            if not final_text.strip():
                return "Document appeared empty after processing (images/icons were ignored)."
                
            print(f"DEBUG: Processing finished. Final text length: {len(final_text)}")
            return final_text

        except Exception as e:
            print(f"DEBUG: MarkItDown conversion error: {e}")
            return f"[Processing Error: {str(e)}]"
