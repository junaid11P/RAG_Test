# RAG SaaS - Intelligent Document Chat

A premium RAG (Retrieval-Augmented Generation) platform for chatting with PDF, TXT, and Word documents.

## 🚀 Features
- **Multi-format Support**: PDF, DOCX, TXT.
- **Glassmorphism UI**: High-end modern design using React & Framer Motion.
- **Smart RAG Pipeline**: Local vector embeddings with FAISS and Groq (Llama 3.3).
- **Usage Tracking**: Real-time billing and usage stats stored in MongoDB.
- **Guest Access**: Free 3-query trial for guest users.

## 🛠️ Tech Stack
- **Backend**: FastAPI, LangChain, FAISS, Motor (MongoDB), Groq.
- **Frontend**: Vite, React, Lucide Icons, Framer Motion.
- **Database**: MongoDB (Async).

## 📂 Project Structure
```text
├── backend/            # FastAPI Application
│   ├── app/            # Main application logic
│   │   ├── api/        # Auth & API routes
│   │   ├── db/         # MongoDB connection
│   │   └── services/   # RAG, LLM, Processor, Usage services
│   ├── uploads/        # User-uploaded documents
│   ├── storage/        # FAISS Vector Indexes
│   └── main.py         # Entry point
└── frontend/           # Vite + React Frontend
    ├── src/            # Components & Logic
    └── index.html      # Entry point
```

## 🛠️ Setup

### Backend
1. `cd backend`
2. `pip install -r requirements.txt`
3. Create `.env` with `GROQ_API_KEY` and `MONGODB_URL`.
4. `python main.py`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`
