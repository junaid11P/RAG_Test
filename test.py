import requests

# 1. Your unique per-document API key from the Dashboard
API_KEY = " "
# 2. Base URL of your RAGI instance
BASE_URL = "http://localhost:8000" 

def ask_ragi(question):
    """
    Query the internal knowledge base of your document.
    """
    url = f"{BASE_URL}/api/v1/query"
    headers = {"X-API-Key": API_KEY}
    params = {"query": question}
    
    try:
        # We send a POST request with the query as a parameter
        response = requests.post(url, headers=headers, params=params)
        response.raise_for_status()
        
        # The API returns JSON with 'query' and 'answer' keys
        data = response.json()
        return data.get("answer")
        
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

# Example: Asking a question
if __name__ == "__main__":
    result = ask_ragi("Give me a summary of this document")
    print(f"RAGI Answer: {result}")