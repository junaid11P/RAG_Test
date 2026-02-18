import re
import os
import io
import uuid
import tempfile
from typing import List, Optional
from app.db.mongodb import db

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

class DocumentProcessor:
    """
    Unified Document Processor using Microsoft MarkItDown.
    Handles PDF, DOCX, DOC, XLSX, PPTX, CSV, HTML, Images, etc.
    Now with automated image extraction and cloud storage.
    """
    _md = MarkItDown() if MarkItDown else None

    @staticmethod
    def basic_clean(text: str) -> str:
        """Cleans text while preserving structural characters like | for tables."""
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
        Extracts images, stores them in GridFS, and embeds Markdown links in text.
        """
        if not cls._md:
            return "[Error: MarkItDown library not installed or initialized.]"

        try:
            # 1. Convert document to Markdown
            result = cls._md.convert(file_path)
            if not result or not result.text_content:
                return "No readable text found in document."

            text_content = result.text_content

            # 2. Extract Images if any (MarkItDown extracts them to a temporary location)
            # We can check for image references in the markdown
            # e.g., ![image](path/to/image.png)
            
            # Simple regex to find image paths in generated markdown
            img_pattern = r"!\[.*?\]\((.*?)\)"
            img_matches = re.findall(img_pattern, text_content)

            for local_img_path in img_matches:
                # If it's a local file path relative to the temporary processing dir
                if os.path.exists(local_img_path):
                    asset_id = str(uuid.uuid4())
                    asset_name = os.path.basename(local_img_path)
                    
                    with open(local_img_path, "rb") as f:
                        file_data = f.read()
                        
                    # Store in media_fs
                    grid_id = await db.media_fs.upload_from_stream(
                        asset_name,
                        file_data,
                        metadata={
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "asset_id": asset_id,
                            "type": "extracted_image"
                        }
                    )
                    
                    # Replace the local path with a RAGI proxy URL that the frontend can load
                    proxy_url = f"/api/media/{asset_id}"
                    text_content = text_content.replace(local_img_path, proxy_url)

            # 3. Clean and Finalize
            clean_text = cls.basic_clean(text_content)
            final_text = cls.remove_noise(clean_text)
            
            if not final_text.strip():
                return "Document appeared empty after processing."
                
            return final_text

        except Exception as e:
            print(f"MarkItDown conversion error: {e}")
            return f"[Processing Error: {str(e)}]"
