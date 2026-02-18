import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # For MVP, we'll allow initialization but warn. 
            # In production, this should raise an error.
            print("Warning: GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "llama-3.3-70b-versatile"

    def generate_answer(self, query: str, context: list):
        """Generates an answer using Groq based on the provided context."""
        if not self.client:
            return "Error: Groq API key not configured. Please add GROQ_API_KEY to your .env file."

        context_text = "\n\n".join(context)
        prompt = f"""
        You are an intelligent assistant. Use the provided context to answer the user's question accurately.
        
        CRITICAL: The context contains Markdown. If the context includes references to images (e.g., `![image](/api/media/...)`) or tables that are relevant to the user's question, you MUST include those exact Markdown references in your answer so the user can see them.
        
        If the answer is not in the context, say that you don't know based on the documents.
        
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
