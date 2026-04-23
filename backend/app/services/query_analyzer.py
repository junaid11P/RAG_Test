import json
import re

class QueryAnalyzer:
    def __init__(self, llm_client, model="llama-3.3-70b-versatile"):
        self.client = llm_client
        self.model = model

    def analyze_query(self, query: str) -> dict:
        """Determines intent, keywords, and time sensitivity."""
        prompt = f"""
        Analyze the following user query and extract:
        1. intent (e.g. search, summarize, compare, explain, lookup, procedural)
        2. keywords (list of strings)
        3. time_sensitive (boolean: true if asking for latest/historical data)
        4. required_aspects (list of strings: DYNAMICALLY INFER THESE based on the domain of the query. Examples: ["payment terms", "late fees"] for invoices, ["obligations", "termination"] for contracts, ["dosage", "side effects"] for medical, etc.)
        5. answer_style (string: "extractive", "analytical", "summary", "procedural")
        6. needs_cross_section_reasoning (boolean: true if it requires combining information from widely different document sections)
        
        Output ONLY a JSON object with this exact structure:
        {{
            "intent": "search",
            "keywords": ["example", "keyword"],
            "time_sensitive": false,
            "required_aspects": [],
            "answer_style": "extractive",
            "needs_cross_section_reasoning": false
        }}
        
        Query: {query}
        """
        fallback = {
            "intent": "search", 
            "keywords": [query], 
            "time_sensitive": False,
            "required_aspects": [],
            "answer_style": "extractive",
            "needs_cross_section_reasoning": False
        }
        
        if not self.client:
            return fallback
            
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            
            parsed = json.loads(content)
            
            q = query.lower().strip()
            if q.startswith("what is ") or q.startswith("what are ") or q.startswith("define ") or q.startswith("who is "):
                parsed["required_aspects"] = []
                parsed["answer_style"] = "extractive"
                parsed["needs_cross_section_reasoning"] = False
                
            # basic validation
            if "intent" not in parsed: parsed["intent"] = "search"
            if "keywords" not in parsed: parsed["keywords"] = [query]
            if "time_sensitive" not in parsed: parsed["time_sensitive"] = False
            if "required_aspects" not in parsed: parsed["required_aspects"] = []
            if "answer_style" not in parsed: parsed["answer_style"] = "extractive"
            if "needs_cross_section_reasoning" not in parsed: parsed["needs_cross_section_reasoning"] = False
            return parsed
        except Exception as e:
            print(f"Query analysis failed: {e}")
            return fallback
