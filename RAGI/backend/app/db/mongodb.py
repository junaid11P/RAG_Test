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

db = MongoDB()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(MONGODB_URL)
    db.db = db.client[DB_NAME]
    db.fs = AsyncIOMotorGridFSBucket(db.db)
    
    # Create Unique Index on Email
    await db.db["users"].create_index("email", unique=True)
    
    print("Connected to MongoDB & GridFS (Unique Email Index ensured)")

async def close_mongo_connection():
    db.client.close()
    print("Closed MongoDB connection")
