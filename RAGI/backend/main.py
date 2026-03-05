from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import os
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

app = FastAPI(title="RAG SaaS API")

# Setup CORS for frontend
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://ragimodels.vercel.app",
    "https://ragimodels-git-main-junaid11ps-projects.vercel.app", # Potential preview URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                
                # 1. Delete Vectors
                await db.db["vectors"].delete_many({"doc_id": doc_id})
                
                # 2. Delete Original File from GridFS
                if file_id:
                    try:
                        from bson import ObjectId
                        await db.fs.delete(ObjectId(file_id))
                    except:
                        pass
                
                # 3. Delete from DB
                await db.db["documents"].delete_one({"id": doc_id})
                print(f"Purged expired doc: {doc['name']}")
                
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        await asyncio.sleep(3600) # Run every hour

# Lifespan events
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()
    import asyncio
    asyncio.create_task(cleanup_expired_docs())

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])

rag_service = RAGService()
llm_service = LLMService()

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        return None
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id
    except (JWTError, IndexError):
        return None

async def get_current_user_required(user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required for this action")
    return user_id

@app.get("/")
async def root():
    return {"message": "RAG SaaS Backend is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    # 1. Validation
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc", ".xlsx", ".pptx", ".csv", ".html"}
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
        metadata={"user_id": effective_user_id, "doc_id": doc_id}
    )

    # 3. Process
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
            
        try:
            clean_text = await DocumentProcessor.process_document(tmp_path, effective_user_id, doc_id)
            await rag_service.create_rag(clean_text, effective_user_id, doc_id)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # 4. Save Metadata ONLY if logged in
        if not is_guest:
            from datetime import timedelta
            doc_meta = {
                "id": doc_id,
                "user_id": user_id,
                "file_id": str(file_id),
                "name": file.filename,
                "file_size": file_size,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=2), # 2 days trial
                "is_premium": False,
                "api_key": None
            }
            await db.db["documents"].insert_one(doc_meta)
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
async def list_documents(user_id: str = Depends(get_current_user_required)):
    cursor = db.db["documents"].find({"user_id": user_id})
    docs = await cursor.to_list(length=100)
    # Convert Mongo _id to string for JSON serialization
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = Depends(get_current_user_required)):
    doc = await db.db["documents"].find_one({"id": doc_id, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Delete from DB
    await db.db["documents"].delete_one({"id": doc_id, "user_id": user_id})
    
    # 2. Delete Vectors
    await db.db["vectors"].delete_many({"doc_id": doc_id, "user_id": user_id})
    
    # 2.5 Delete Chat History
    await db.db["chat_history"].delete_many({"doc_id": doc_id, "user_id": user_id})
    
    # 3. Delete from GridFS
    file_id = doc.get("file_id")
    file_size = doc.get("file_size", 0)
    
    if file_id:
        from bson import ObjectId
        try:
            await db.fs.delete(ObjectId(file_id))
        except:
            pass
    
    # 4. Update Usage Stats
    await UsageService.track_delete(user_id, file_size)
    
    return {"status": "success", "message": "Document, vectors, and chat history purged permanently from Atlas"}

@app.get("/documents/{doc_id}/history")
async def get_document_history(doc_id: str, user_id: str = Depends(get_current_user)):
    identity = user_id if user_id else None # We only persist for logged in users in this logic
    if not identity:
        return []
        
    history = await ChatService.get_history(identity, doc_id)
    return history

@app.post("/documents/{doc_id}/api-key")
async def generate_api_key(doc_id: str, user_id: str = Depends(get_current_user_required)):
    new_key = f"RAGI_{uuid.uuid4().hex}"
    res = await db.db["documents"].update_one(
        {"id": doc_id, "user_id": user_id},
        {"$set": {"api_key": new_key}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "api_key": new_key}

@app.post("/documents/{doc_id}/upgrade")
async def upgrade_to_premium(doc_id: str, user_id: str = Depends(get_current_user_required)):
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

    # Check if doc exists in vectors
    doc_exists = await db.db["vectors"].find_one({"doc_id": doc_id, "user_id": user_id})
    if not doc_exists:
        raise HTTPException(status_code=404, detail="Document index not found in Atlas.")

    context = await rag_service.query_rag(doc_id, query, user_id)
    answer = llm_service.generate_answer(query, context)
    
    # Save to history & usage
    await ChatService.save_message(user_id, doc_id, "user", f"[API] {query}")
    await ChatService.save_message(user_id, doc_id, "system", answer)
    await UsageService.track_api_call(user_id)
    
    return {
        "query": query,
        "answer": answer,
        "doc_name": doc["name"]
    }

@app.post("/query")
async def query_document(doc_id: str, query: str, user_id: str = Depends(get_current_user), guest_id: str = None):
    # Determine the effective identity
    identity = user_id if user_id else guest_id
    
    if not identity:
        raise HTTPException(status_code=401, detail="Authentication or Guest session required")

    # If it's a guest, check the 3-query limit
    is_guest = user_id is None
    if is_guest:
        # Increment and get count
        q_count = await UsageService.get_guest_query_count(identity)
        if q_count > 3:
            # Purge Guest Data from Atlas
            await db.db["vectors"].delete_many({"user_id": identity})
            
            # Find and delete guest files from GridFS
            cursor = db.fs.find({"metadata.user_id": identity})
            async for grid_out in cursor:
                await db.fs.delete(grid_out._id)
            
            raise HTTPException(
                status_code=403, 
                detail="Free trial limit reached. Guest data has been purged from Atlas. Please login to continue."
            )

    # Check if doc exists in vectors
    doc_exists = await db.db["vectors"].find_one({"doc_id": doc_id, "user_id": identity})
    if not doc_exists:
        raise HTTPException(status_code=404, detail="Document index not found in Atlas for this session.")

    context = await rag_service.query_rag(doc_id, query, identity)
    answer = llm_service.generate_answer(query, context)
    
    # Save to history if logged in
    if not is_guest:
        await ChatService.save_message(user_id, doc_id, "user", query)
        await ChatService.save_message(user_id, doc_id, "system", answer)
        await UsageService.track_api_call(user_id)
    
    return {
        "query": query,
        "answer": answer,
        "is_temporary": is_guest,
        "queries_remaining": max(0, 3 - q_count) if is_guest else "Unlimited"
    }

@app.get("/usage")
async def get_usage(user_id: str = Depends(get_current_user_required)):
    usage_stats = await UsageService.get_usage(user_id)
    if not usage_stats:
        raise HTTPException(status_code=404, detail="User not found")
    return usage_stats

@app.post("/api/payments/verify")
async def submit_payment_verification(
    data: dict, 
    user_id: str = Depends(get_current_user_required)
):
    utr = data.get("utr_number")
    email = data.get("email")
    v_type = data.get("type", "upgrade")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    vid = await PaymentService.submit_verification(user_id, utr, email, type=v_type)
    return {"status": "success", "verification_id": vid, "message": "Verification request submitted successfully"}

@app.get("/api/payments/status")
async def get_payment_status(user_id: str = Depends(get_current_user_required)):
    verifications = await PaymentService.get_user_verifications(user_id)
    return verifications

if __name__ == "__main__":
    import uvicorn
    # Use $PORT from environment (default to 8000 for local)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
