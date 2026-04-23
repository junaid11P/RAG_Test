import os
import logging
from groq import Groq
from dotenv import load_dotenv

import json

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # For MVP, we'll allow initialization but warn. 
            # In production, this should raise an error.
            logging.warning("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.3-70b-versatile"

    def generate_answer(self, query: str, context: list):
        """Generates an answer using Groq based on the provided context.
           Context can be a list of strings or a list of dicts with 'text' and 'confidence_score'.
        """
        if not self.client:
            return "Error: Groq API key not configured. Please add GROQ_API_KEY to your .env file."

        # Process context based on type (string vs dict with scores)
        formatted_contexts = []
        for i, ctx in enumerate(context):
            if isinstance(ctx, dict):
                score_pct = round(ctx.get('confidence_score', 0) * 100, 2)
                formatted_contexts.append(f"[Source {i+1} | Confidence: {score_pct}%]\n{ctx.get('text', '')}")
            else:
                formatted_contexts.append(f"[Source {i+1}]\n{ctx}")
                
        context_text = "\n\n".join(formatted_contexts)
        
        prompt = f"""
        You are an intelligent assistant. Use the provided context to answer the user's question accurately.
        
        CRITICAL: 
        1. The context contains Markdown. If the context includes tables that are relevant to the user's question, you MUST preserve the table formatting in your answer.
        2. Ground your answer EXPLICITLY in the provided context. Cite source numbers if possible.
        3. If the answer is not in the context, say that you don't know based on the documents.
        
        Context:
        {context_text}
        
        User Question: {query}
        
        Answer:
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"


