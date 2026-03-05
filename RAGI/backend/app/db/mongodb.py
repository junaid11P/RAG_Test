from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://junaid11:Pass2025@cluster0.xbznqfp.mongodb.net/rag_saas?appName=Cluster0")
DB_NAME = "rag_saas"

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None
    fs = None
    media_fs = None

db = MongoDB()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(MONGODB_URL)
    db.db = db.client[DB_NAME]
    db.fs = AsyncIOMotorGridFSBucket(db.db)
    db.media_fs = AsyncIOMotorGridFSBucket(db.db, bucket_name="media")
    
    # Standard Index: Unique Email for Users
    await db.db["users"].create_index("email", unique=True)

    # TTL Indexes: Automatic Deletion after 24 Hours (86400 seconds)
    # 1. Delete expired documents based on 'expires_at' field
    # Documents with 'expires_at' set will be deleted when that time is reached.
    await db.db["documents"].create_index("expires_at", expireAfterSeconds=0)
    
    # 2. Delete guest sessions after 24 hours of creation
    await db.db["guest_sessions"].create_index("created_at", expireAfterSeconds=86400)
    
    # 3. Delete document embeddings (vectors) if they have an 'expires_at' field
    await db.db["document_embeddings"].create_index("expires_at", expireAfterSeconds=0)

    # 4. Delete conversation history if it has an 'expires_at' field
    await db.db["conversation_history"].create_index("expires_at", expireAfterSeconds=0)
    
    print("Connected to MongoDB & GridFS (Main + Media)")
    print("TTL Indexes initialized: Guest sessions and expired documents will be auto-purged.")

async def close_mongo_connection():
    db.client.close()
    print("Closed MongoDB connection")
