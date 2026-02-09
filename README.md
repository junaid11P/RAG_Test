# RAG Story Assistant (Milvus + Ollama)

A local Retrieval-Augmented Generation (RAG) pipeline designed to search through stories and answer questions using a vector database (Milvus) and a local LLM (Ollama).

## 🚀 Features
- **Local Vector Search**: Powered by Milvus for fast, high-dimensional similarity search.
- **Local LLM**: Uses Ollama with `llama3.2` for private and secure response generation.
- **Automated Ingestion**: Scripts to chunk, embed, and insert story data.
- **Visual Management**: Includes Attu for easy monitoring of your Milvus collections.

---

## 🛠️ Prerequisites
- **Docker & Docker Compose**
- **Python 3.10+**
- **Ollama** installed and running (`ollama serve`)

---

## ⚙️ Setup

### 1. Start Infrastructure
Run the following command to start Milvus, Etcd, Minio, and Attu:
```bash
docker compose up -d
```
- **Milvus**: Listening on `localhost:19530`
- **Attu (UI)**: Access at [http://localhost:8000](http://localhost:8000)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Ollama
Ensure you have the model pulled:
```bash
ollama pull llama3.2
```

---

## 📂 Project Structure
- `ingest_data.py`: Reads `story.txt` and pushes chunks to Milvus `story_chunks` collection.
- `milvus_search.py`: Module for performing vector search on stored embeddings.
- `rag_answer.py`: The main entry point to ask questions and get AI-generated answers.
- `chunk.py` / `read.py`: Utility scripts for text processing.
- `docker-compose.yml`: Infrastructure configuration.

---

## 🏃 Usage

### Step 1: Ingest Data
Prepare your story text in `story.txt`, then run:
```bash
python ingest_data.py
```

### Step 2: Search & Answer
Ask questions about the story:
```bash
python rag_answer.py
```

---

## 🔍 Visualizing Data
You can use **Attu** to view your collections, search scores, and raw text.  
Simply visit **[http://localhost:8000](http://localhost:8000)** and connect to `milvus-standalone:19530`.
