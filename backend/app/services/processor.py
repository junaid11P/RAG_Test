import re
import os
import io
import uuid
import base64
import logging
from typing import List, Optional
from app.db.mongodb import db
from PIL import Image

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None

class DocumentProcessor:
    @classmethod
    def get_md_engine(cls):
        """Initializes MarkItDown with vision capabilities using Groq LLM to parse images."""
        if not MarkItDown:
            return None
            
        import os
        from groq import Groq
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key:
            client = Groq(api_key=api_key)
            # Use Llama 4 Scout multimodal model for high-fidelity visual understanding
            return MarkItDown(llm_client=client, llm_model="meta-llama/llama-4-scout-17b-16e-instruct")
        else:
            return MarkItDown()

    @staticmethod
    def basic_clean(text: str) -> str:
        """Cleans text while preserving structural characters like | for tables."""
        # Remove common icon/image placeholders that might be left as text
        text = re.sub(r"\[image\]|\[icon\]|\[pic\]", "", text, flags=re.IGNORECASE)
        # Safer text cleaning: preserve unicode characters
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def remove_noise(text: str) -> str:
        """Removes common digital noise without destroying table structure."""
        noise_patterns = [
            r"mailto:.*",
            r"copyright\s+©\s+\d{4}",
            r"page\s+\d+\s+of\s+\d+",
            r"(?i)confidential"
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    async def _extract_pdf_visuals(cls, file_path: str) -> str:
        """
        Manually extracts images from PDFs and uses Groq Vision to describe them.
        This compensates for MarkItDown missing images in certain PDF structures.
        """
        if not fitz:
            return ""

        import os
        from groq import Groq
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return ""

        client = Groq(api_key=api_key)
        all_descriptions = []

        try:
            doc = fitz.open(file_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Filter small icons/logos (under 10KB) to save tokens
                    if len(image_bytes) < 10000:
                        continue
                        
                    # Process image for Groq Vision
                    img_io = io.BytesIO(image_bytes)
                    image = Image.open(img_io)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Compress slightly to ensure we stay under 4MB limit
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    try:
                        resp = client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Extract and summarize all text, charts, and data science information from this image. Be precise and grounding."},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                                        },
                                    ],
                                }
                            ],
                            max_tokens=600
                        )
                        desc = resp.choices[0].message.content
                        all_descriptions.append(f"[Image Description (Page {page_index+1})]: {desc}")
                    except Exception as e:
                        logging.error(f"Groq Vision error on page {page_index+1}: {e}")
            
            doc.close()
        except Exception as e:
            logging.error(f"PyMuPDF processing error: {e}")

        return "\n\n".join(all_descriptions) if all_descriptions else ""

    @classmethod
    async def process_document(cls, file_path: str, user_id: str, doc_id: str) -> str:
        """
        Complete production pipeline using MarkItDown.
        Uses AI Vision to OCR images/icons and explicitly preserves text and tables.
        Runs in a separate thread to prevent blocking the event loop.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        md = cls.get_md_engine()
        if not md:
            return "[Error: MarkItDown library not installed.]"

        try:
            logging.info(f"Starting conversion for {file_path}...")
            # 1. Convert document to Markdown (Run in thread to prevent blocking)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, md.convert, file_path)
            
            if not result or not result.text_content:
                return "No readable text found in document."
 
            text_content = result.text_content
            logging.info(f"Conversion complete. Content length: {len(text_content)}")
 
            # 2. Format Image Descriptions from MarkItDown
            # We strip binary but keep descriptions
            text_content = re.sub(r"!\[(.*?)\]\(.*?\)", r"[Image Description: \1]", text_content)
 
            # 3. Handle PDF Visuals specifically if it's a PDF
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".pdf":
                logging.info(f"Running advanced PDF visual extraction for {file_path}...")
                visual_context = await cls._extract_pdf_visuals(file_path)
                if visual_context:
                    text_content += "\n\n" + visual_context
            
            # 4. Clean and Finalize
            clean_text = cls.basic_clean(text_content)
            final_text = cls.remove_noise(clean_text)
            
            if not final_text.strip():
                return "Document appeared empty after processing."
                
            logging.info(f"Processing finished. Final text length: {len(final_text)}")
            return final_text
 
        except Exception as e:
            logging.error(f"MarkItDown conversion error: {e}", exc_info=True)
            return f"[Processing Error: {str(e)}]"
