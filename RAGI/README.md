# RAGI - Premium Intelligent Document Chat

A high-performance RAG (Retrieval-Augmented Generation) platform designed for seamless interaction with PDF, TXT, and Word documents. Built with a focus on speed, aesthetics, and scalability.

---

## 🚀 Key Features

- **Multi-format Support**: Intelligent extraction from PDF, DOCX, and TXT files.
- **Ultra-Light RAG Pipeline**: Powered by **FastEmbed (ONNX)** for high-speed embeddings without the overhead of heavy deep learning frameworks.
- **Glassmorphism UI**: A stunning, modern design built with **React** and **Framer Motion** for a premium user experience.
- **Cloud Vector Search**: Leverages **MongoDB Atlas Vector Search** for scalable, zero-latency document retrieval.
- **Guest Access**: Instant value with a 3-query trial for guest users; permanent storage for registered users.
- **Real-time Analytics**: Built-in usage tracking and quota management.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (High-performance Python framework)
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
│   │   ├── api/        # Authentication & API Endpoints
│   │   ├── core/       # Security & Config
│   │   ├── db/         # MongoDB Connectivity (GridFS & Atlas)
│   │   └── services/   # Business Logic (RAG, LLM, Processors)
│   ├── main.py         # Application Entry Point
│   └── requirements.txt
└── frontend/           # React Application
    ├── src/
    │   ├── components/ # Reusable UI Components
    │   ├── pages/      # View Modules
    │   └── theme/      # Global Design System
    └── package.json
```

---

## ⚙️ Configuration & Deployment

### Environment Variables
Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URL=your_mongodb_atlas_connection_string
SECRET_KEY=your_jwt_secret
```

### Setup Instructions

#### 1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

#### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## ☁️ Deployment Notes (Render.com)

This project is optimized for **Render's Free Tier (512MB RAM)**:
- **FastEmbed** is used instead of PyTorch to keep memory usage < 200MB.
- **GridFS** handles files in the cloud, removing the need for persistent disk storage.
- **Stateless Architecture**: Perfect for serverless or ephemeral container deployments.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
