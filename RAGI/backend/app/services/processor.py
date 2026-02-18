import re
import os
import io
import uuid
from typing import List, Optional
from app.db.mongodb import db

try:
    from markitdown import MarkItDown
    from openai import OpenAI
except ImportError:
    MarkItDown = None
    OpenAI = None

class DocumentProcessor:
    """
    Unified Document Processor using Microsoft MarkItDown with Groq Vision.
    Handles PDF, Office, and Raw Images with Intelligent OCR.
    """
    
    @classmethod
    def get_md_engine(cls):
        """Initializes MarkItDown with Groq Vision capabilities if API key exists."""
        if not MarkItDown:
            return None
            
        api_key = os.getenv("GROQ_API_KEY")
        if OpenAI and api_key:
            # We use the OpenAI-compatible Groq endpoint to bridge with MarkItDown
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key
            )
            # Use Llama 3.2 Vision for high-quality image-to-text conversion
            return MarkItDown(llm_client=client, llm_model="llama-3.2-11b-vision-preview")
        
        return MarkItDown()

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
        Complete production pipeline using MarkItDown + Groq Vision.
        Extracts images, OCRs them using LLM, and stores originals in GridFS.
        """
        md = cls.get_md_engine()
        if not md:
            return "[Error: MarkItDown library not installed.]"

        try:
            # 1. Convert document to Markdown (MarkItDown uses the LLM to describe images found)
            result = md.convert(file_path)
            if not result or not result.text_content:
                return "No readable text found in document."

            text_content = result.text_content

            # 2. Extract and Proxy Images
            # We still want to see the original images in the chat, 
            # while the text context contains the LLM's OCR/description.
            img_pattern = r"!\[.*?\]\((.*?)\)"
            img_matches = re.findall(img_pattern, text_content)

            for local_img_path in img_matches:
                # If it's a local file path (MarkItDown saves images to temp files during conversion)
                if os.path.exists(local_img_path):
                    asset_id = str(uuid.uuid4())
                    asset_name = os.path.basename(local_img_path)
                    
                    with open(local_img_path, "rb") as f:
                        file_data = f.read()
                        
                    # Store original image in media_fs
                    await db.media_fs.upload_from_stream(
                        asset_name,
                        file_data,
                        metadata={
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "asset_id": asset_id,
                            "type": "extracted_image"
                        }
                    )
                    
                    # Replace the local path with a RAGI proxy URL
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
