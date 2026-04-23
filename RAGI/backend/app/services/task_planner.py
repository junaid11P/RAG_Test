import json
import re

class TaskPlanner:
    def __init__(self, llm_client, model="llama-3.3-70b-versatile"):
        self.client = llm_client
        self.model = model

    def generate_plan(self, query: str, analyzer_output: dict = None) -> dict:
        """
        Takes Query Analyzer output, breaks query into steps, decides retrieval strategy.
        Outputs: {"steps": [], "retrieval_type": "semantic/hybrid", "top_k": 5}
        """
        analyzer_output_str = json.dumps(analyzer_output) if analyzer_output else "{}"
        prompt = f"""
        You are a Task Planner Agent for a Retrieval-Augmented Generation system.
        The user has asked a query. The Query Analyzer has provided this context about the query:
        {analyzer_output_str}
        
        Break down the user's query into specific steps to retrieve the required information.
        Decide the retrieval strategy: use "semantic" for general focus, "hybrid" if specific keyword matching is critical.
        Determine how many top chunks (top_k) should be retrieved per step. (Usually 3-5).
        
        Output ONLY a valid JSON object with this exact structure:
        {{
            "steps": [
                {{"query": "extract demographic data", "aspect": "demographics", "rationale": "Identify the base population"}},
                {{"query": "extract payment term exceptions", "aspect": "exemptions", "rationale": "Find if there are waivers for late fees"}}
            ],
            "retrieval_type": "semantic",
            "top_k": 5
        }}
        
        Adapt the "aspect" field to the domain-specific topics required.
        
        User Question: {query}
        """
        
        fallback = {
            "steps": [{"query": query, "aspect": "general", "rationale": "Default single step retrieval"}], 
            "retrieval_type": "semantic", 
            "top_k": 5
        }
        
        if not self.client:
            return fallback

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            
            parsed = json.loads(content)
            if "steps" not in parsed: 
                parsed["steps"] = [{"query": query, "aspect": "general", "rationale": "Direct query"}]
            if "retrieval_type" not in parsed: parsed["retrieval_type"] = "semantic"
            if "top_k" not in parsed: parsed["top_k"] = 5
            
            # Format string steps to dicts if model ignored schema
            for i, step in enumerate(parsed["steps"]):
                if isinstance(step, str):
                    parsed["steps"][i] = {"query": step, "aspect": "general", "rationale": "Direct query step"}
            
            return parsed
            
        except Exception as e:
            print(f"Task planning failed: {e}")
            return fallback
