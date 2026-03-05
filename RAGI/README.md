# RAGI - Premium Intelligent Document Chat & API Platform

**RAGI** (Retrieval-Augmented Generation Interface) is a high-performance RAG platform designed for seamless interaction with a wide array of document formats. It goes beyond simple chat by allowing users to transform their documents into **queryable APIs**.

---

## 🌟 The "Main Event": Contextual API Integration
The core differentiator of RAGI is its **Developer-First Architecture**. Once a user registers and uploads a document, they can:
1. **Generate a Unique API Key**: Specifically bound to that document's context.
2. **External Integration**: Query the document's knowledge base from any third-party application (Python, JavaScript, cURL, etc.) using a secure REST endpoint.
3. **Persistent Knowledge**: Manage multiple document "models" and their respective keys from a central dashboard.

---

## 🚀 Key Features

- **Contextual API Keys**: Generate per-document keys to integrate RAG capabilities into your own apps.
- **Multi-format Support**: Intelligent extraction from PDF, DOCX, XLSX, PPTX, CSV, and  using **Microsoft MarkItDown**.
- **Ultra-Light RAG Pipeline**: Powered by **FastEmbed (ONNX)** for high-speed embeddings with minimal memory overhead.
- **Glassmorphism UI**: A stunning, modern design built with **React** and **Framer Motion** for a premium user experience.
- **Cloud Vector Search**: Leverages **MongoDB Atlas Vector Search** for scalable, zero-latency document retrieval.
- **Flexible Access**: 
  - **Guest Access**: Instant 3-query trial (ephemeral storage).
  - **Registered Access**: Permanent document storage, chat history, and API key management.
- **Real-time Analytics**: Built-in usage tracking and quota management.

---

## 🛠️ How It Works (Working Steps)

### 1. Registration & Authentication
Create an account to unlock permanent storage. While guests can trial the system, registered users get access to the dedicated **API Dashboard** and document persistence.

### 2. Intelligent Document Processing
Upload any supported file (PDF, Office,).
- **Extraction**: Microsoft **MarkItDown** cleans and structured the text.
- **Vectorization**: **FastEmbed** generates high-dimensional embeddings.
- **Storage**: Vectors are pushed to **MongoDB Atlas** for semantic search.

### 3. API Key Generation
Navigate to your document list and click **"Generate Key"**. This creates a unique `RAGI_` prefix key specifically tied to that document's vector index.

### 4. Integration
Copy your API Key and use the `/api/v1/query` endpoint to build your own chatbots, automation scripts, or enterprise tools powered by your document's data.

---

## 💻 Tech Stack

### Backend
- **Framework**: FastAPI (High-performance Python framework)
- **Processing**: Microsoft MarkItDown (Universal document-to-markdown conversion)
- **Orchestration**: LangChain (Modular RAG architecture)
- **Embeddings**: FastEmbed (Optimized ONNX Runtime - Ultra-low memory footprint)
- **LLM**: Groq (Llama 3.3 70B - Blazing fast inference)
- **Database**: MongoDB (Motor / GridFS / Atlas Vector Search)

### Frontend
- **Framework**: Vite + React
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Styling**: Modern CSS with Glassmorphism tokens

---

## 📂 Project Structure

```text
├── backend/            # FastAPI Application
│   ├── app/
│   │   ├── api/        # Auth & API Routes
│   │   ├── core/       # Security (JWT) & Config
│   │   ├── db/         # MongoDB Connectivity (GridFS & Vector Search)
│   │   ├── models/     # Data Models
│   │   └── services/   # RAG Logic, LLM, Processors, & Usage Tracking
│   ├── main.py         # App Entry Point & API Endpoints
│   └── requirements.txt
└── frontend/           # React Application
    ├── src/
    │   ├── components/ # Reusable UI Components
    │   ├── pages/      # View Modules (Chat, Docs, API Keys)
    │   └── theme/      # Global Design System
    └── package.json
```

---

## ⚙️ Configuration & Deployment

### 1. Environment Variables
Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URL=your_mongodb_atlas_connection_string
SECRET_KEY=your_jwt_secret
```

### 2. Setup Instructions

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py  # Automatically detects host and port
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## ☁️ Deployment Guide

### Backend (Render.com)
1. **GitHub**: Push the `backend/` folder.
2. **Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
   - **Env Vars**: Add `MONGODB_URL`, `GROQ_API_KEY`, and `SECRET_KEY`.
3. **Dynamic Port**: The code automatically detects the Render `$PORT`.

### Frontend (Vercel.com)
1. **GitHub**: Push the `frontend/` folder.
2. **Settings**:
   - **Framework**: Vite.
   - **Env Vars**: Set `VITE_API_BASE` to your Render backend URL.
3. **SPA Mode**: Enabled via `vercel.json` already in the repo.



## 🔗 Developer API Example (Python)

```python
import requests

API_KEY = "YOUR_RAGI_DOCUMENT_KEY"
URL = "http://localhost:8000/api/v1/query"

response = requests.post(
    URL, 
    headers={"X-API-Key": API_KEY},
    params={"query": "What are the key findings in this document?"}
)

print(response.json()["answer"])
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

