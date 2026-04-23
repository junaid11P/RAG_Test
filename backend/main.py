from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
import logging
import traceback
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import os

# Fix HuggingFace/FastEmbed symlink privileges error on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import uuid
import asyncio
from datetime import datetime
from app.db.mongodb import db

from app.services.processor import DocumentProcessor
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.usage_service import UsageService
from app.services.chat_service import ChatService
from app.services.payment_service import PaymentService
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.api.auth import router as auth_router
from app.core.security import SECRET_KEY, ALGORITHM
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup DB Client
    await connect_to_mongo()
    cleanup_task = asyncio.create_task(cleanup_expired_docs())
    yield
    # Shutdown DB Client
    cleanup_task.cancel()
    await close_mongo_connection()

app = FastAPI(title="RAG SaaS API", lifespan=lifespan)

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"FATAL ERROR: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Server Error", "message": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

async def purge_document_data(doc_id: str, file_id: str = None, user_id: str = None):
    """Helper to cleanly purge all artifacts related to a document."""
    try:
        # 1. Delete Embeddings and History
        query = {"doc_id": doc_id}
        if user_id:
            query["user_id"] = user_id
            
        await db.db["document_embeddings"].delete_many(query)
        await db.db["conversation_history"].delete_many(query)
        
        # 2. Delete Original File from GridFS
        if file_id:
            try:
                from bson import ObjectId
                await db.fs.delete(ObjectId(file_id))
            except Exception as e:
                logging.warning(f"Failed to delete GridFS file {file_id} for doc {doc_id}: {e}")
                
        # 3. Delete from documents collection
        await db.db["documents"].delete_one(query)
        return True
    except Exception as e:
        logging.error(f"Error purging document {doc_id}: {e}")
        return False

async def cleanup_expired_docs():
    """Background task to delete expired files and DB entries every hour."""
    while True:
        try:
            now = datetime.utcnow()
            expired_docs = db.db["documents"].find({
                "expires_at": {"$lt": now},
                "is_premium": False
            })
            
            async for doc in expired_docs:
                doc_id = doc["id"]
                file_id = doc.get("file_id")
                
                success = await purge_document_data(doc_id, file_id)
                if success:
                    logging.info(f"Purged expired doc: {doc.get('name', doc_id)}")
                    
        except Exception as e:
            logging.error(f"Cleanup error during batch processing: {e}")
            
        # Ensure we always sleep and loop, even if earlier code fails
        try:
            await asyncio.sleep(3600) # Run every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in cleanup sleep: {e}")
            await asyncio.sleep(60) # Fallback sleep to avoid tight loop on error

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

rag_service = RAGService()
llm_service = LLMService()

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        return None
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        email = payload.get("email")
        if not user_id:
            return None
        return {"user_id": user_id, "email": email}
    except (JWTError, IndexError):
        return None

async def get_current_user_required(user = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required for this action")
    return user

@app.get("/")
async def root():
    return {"message": "RAG SaaS Backend is running", "version": "2.0.0"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user = Depends(get_current_user)):
    user_id = user["user_id"] if user else None
    email = user["email"] if user else None

    # 1. Validation
    allowed_extensions = {
        ".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", 
        ".csv", ".json", ".xml", ".md", ".html", ".htm",
        ".jpg", ".jpeg", ".png", ".bmp", ".wav", ".mp3", ".m4a",
        ".zip", ".epub"
    }
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(allowed_extensions)}")

    # 2. Save File to GridFS
    is_guest = user_id is None
    effective_user_id = user_id if user_id else f"guest_{uuid.uuid4().hex[:8]}"
    doc_id = str(uuid.uuid4())
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Store in GridFS
    file_id = await db.fs.upload_from_stream(
        file.filename,
        file_content,
        metadata={"user_id": effective_user_id, "email": email, "doc_id": doc_id}
    )

    # 3. Process
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
            
        try:
            clean_text = await DocumentProcessor.process_document(tmp_path, effective_user_id, doc_id)
            # Create RAG will be called after saving metadata now to ensure expires_at is consistent
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # 4. Save Metadata (Always, including guests)
        from datetime import timedelta
        expiry = datetime.utcnow() + (timedelta(days=2) if not is_guest else timedelta(days=1))
        
        now = datetime.utcnow()
        doc_meta = {
            "id": doc_id,
            "user_id": user_id if user_id else effective_user_id,
            "email": email,
            "file_id": str(file_id),
            "name": file.filename,
            "file_size": file_size,
            "created_at": now,
            "created_date": now.strftime("%Y-%m-%d"),
            "created_time": now.strftime("%H:%M:%S"),
            "expires_at": expiry,
            "is_premium": False,
            "is_guest": is_guest,
            "api_key": None
        }
        await db.db["documents"].insert_one(doc_meta)
        
        # 5. Track Usage and Create RAG with Expiry
        await rag_service.create_rag(clean_text, effective_user_id, doc_id, email=email, expires_at=expiry)
        
        if not is_guest:
            await UsageService.track_upload(user_id, file_size)
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "name": file.filename,
            "user_id": effective_user_id,
            "is_temporary": is_guest,
            "message": "Document processed and stored in Atlas."
        }
    except Exception as e:
        # Cleanup GridFS on failure
        await db.fs.delete(file_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    cursor = db.db["documents"].find({"user_id": user_id})
    docs = await cursor.to_list(length=100)
    # Convert Mongo _id to string for JSON serialization
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    doc = await db.db["documents"].find_one({"id": doc_id, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_id = doc.get("file_id")
    file_size = doc.get("file_size", 0)
    
    success = await purge_document_data(doc_id, file_id, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to completely purge document")
    
    # 4. Update Usage Stats
    await UsageService.track_delete(user_id, file_size)
    
    return {"status": "success", "message": "Document, embeddings, and conversation history purged permanently from Atlas"}

@app.get("/documents/{doc_id}/history")
async def get_document_history(doc_id: str, user = Depends(get_current_user)):
    user_id = user["user_id"] if user else None
    identity = user_id if user_id else None # We only persist for logged in users in this logic
    if not identity:
        return []
        
    history = await ChatService.get_history(identity, doc_id)
    return history

@app.post("/documents/{doc_id}/api-key")
async def generate_api_key(doc_id: str, user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    new_key = f"RAGI_{uuid.uuid4().hex}"
    res = await db.db["documents"].update_one(
        {"id": doc_id, "user_id": user_id},
        {"$set": {"api_key": new_key}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "api_key": new_key}

@app.post("/documents/{doc_id}/upgrade")
async def upgrade_to_premium(doc_id: str, user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    """Simulate payment and make the document permanent."""
    res = await db.db["documents"].update_one(
        {"id": doc_id, "user_id": user_id},
        {"$set": {"is_premium": True, "expires_at": None}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": "Document upgraded to Premium. Expiration removed."}

@app.post("/api/v1/query")
async def external_query(query: str, x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header missing")
    
    # Find document by API Key
    doc = await db.db["documents"].find_one({"api_key": x_api_key})
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    doc_id = doc["id"]
    user_id = doc["user_id"]

    # Check if doc exists in embeddings
    doc_exists = await db.db["document_embeddings"].find_one({"doc_id": doc_id, "user_id": user_id})
    if not doc_exists:
        raise HTTPException(status_code=404, detail="Document index not found in Atlas.")

    from app.services.stca_graph import STCAGraph
    stca_graph = STCAGraph(llm_service, rag_service)
    
    result = await stca_graph.execute(query, doc_id, user_id)
    answer = result.get("answer", "")
    
    # Save to history & usage
    await ChatService.save_message(user_id, doc_id, "user", f"[API] {query}", email=doc.get("email"), expires_at=doc.get("expires_at"))
    await ChatService.save_message(user_id, doc_id, "system", answer, email=doc.get("email"), expires_at=doc.get("expires_at"))
    await UsageService.track_api_call(user_id, email=doc.get("email"))
    
    return {
        "query": query,
        "answer": answer,
        "doc_name": doc["name"],
        "confidence_score": result.get("confidence_score"),
        "sources": result.get("sources"),
        "reasoning": result.get("reasoning"),
        "source_type": result.get("source_type"),
        "note": result.get("note")
    }



@app.post("/query")
async def query_document(doc_id: str, query: str, user = Depends(get_current_user), guest_id: str = None):
    # Determine the effective identity
    user_id = user["user_id"] if user else None
    email = user["email"] if user else None
    identity = user_id if user_id else guest_id
    
    if not identity:
        raise HTTPException(status_code=401, detail="Authentication or Guest session required")

    # If it's a guest, check the 3-query limit
    is_guest = user_id is None
    if is_guest:
        # Increment and get count
        q_count = await UsageService.get_guest_query_count(identity)
        if q_count > 5:
            # Purge Guest Data from Atlas
            await db.db["document_embeddings"].delete_many({"user_id": identity})
            
            # Delete associated documents record
            await db.db["documents"].delete_many({"user_id": identity, "is_guest": True})
            
            # Find and delete guest files from GridFS
            cursor = db.fs.find({"metadata.user_id": identity})
            async for grid_out in cursor:
                await db.fs.delete(grid_out._id)
            
            raise HTTPException(
                status_code=403, 
                detail="Free trial limit reached. Guest data has been purged from Atlas. Please login to continue."
            )

    # Check if doc exists and get expiry
    doc = await db.db["documents"].find_one({"id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or expired.")
    
    expires_at = doc.get("expires_at")

    # Check if doc exists in embeddings
    doc_exists = await db.db["document_embeddings"].find_one({"doc_id": doc_id, "user_id": identity})
    if not doc_exists:
        raise HTTPException(status_code=404, detail="Document index not found in Atlas for this session.")

    # Instantiate LangGraph STCA-RAG
    from app.services.stca_graph import STCAGraph
    stca_graph = STCAGraph(llm_service, rag_service)
    
    # Execute Pipeline
    result = await stca_graph.execute(query, doc_id, identity)
    
    answer = result.get("answer", "")
    
    # Save to history if logged in
    if not is_guest:
        await ChatService.save_message(user_id, doc_id, "user", query, email=email, expires_at=expires_at)
        await ChatService.save_message(user_id, doc_id, "system", answer, email=email, expires_at=expires_at)
        await UsageService.track_api_call(user_id, email=email)
    
    # Add frontend compatibility fields
    result["is_temporary"] = is_guest
    result["queries_remaining"] = max(0, 5 - q_count) if is_guest else "Unlimited"
    
    return result

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "STCA-RAG"}

@app.get("/usage")
async def get_usage(user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    usage_stats = await UsageService.get_usage(user_id)
    if not usage_stats:
        raise HTTPException(status_code=404, detail="User not found")
    return usage_stats

@app.post("/api/payments/verify")
async def submit_payment_verification(
    data: dict, 
    user = Depends(get_current_user_required)
):
    user_id = user["user_id"]
    utr = data.get("utr_number")
    email = data.get("email")
    v_type = data.get("type", "upgrade")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    vid = await PaymentService.submit_verification(user_id, utr, email, type=v_type)
    return {"status": "success", "verification_id": vid, "message": "Verification request submitted successfully"}

@app.get("/api/payments/status")
async def get_payment_status(user = Depends(get_current_user_required)):
    user_id = user["user_id"]
    verifications = await PaymentService.get_user_verifications(user_id)
    return verifications

if __name__ == "__main__":
    import uvicorn
    # Render requires listening on the $PORT environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
