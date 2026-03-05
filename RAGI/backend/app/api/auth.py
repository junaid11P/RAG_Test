from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from app.db.mongodb import db
from app.core.security import get_password_hash, verify_password, create_access_token
from datetime import datetime
import uuid

router = APIRouter()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

@router.post("/register", response_model=UserResponse)
async def register(user_in: UserCreate):
    try:
        # Check if user exists
        existing_user = await db.db["users"].find_one({"email": user_in.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_id = str(uuid.uuid4())
        print(f"DEBUG: Registering user {user_in.email} with ID {user_id}")
        
        user_dict = {
            "id": user_id,
            "email": user_in.email,
            "hashed_password": get_password_hash(user_in.password),
            "created_at": datetime.utcnow(),
            "usage": {
                "files_uploaded": 0,
                "api_calls": 0,
                "total_bytes": 0
            }
        }
        
        await db.db["users"].insert_one(user_dict)
        print("DEBUG: User inserted successfully")
        return UserResponse(id=user_id, email=user_in.email, created_at=user_dict["created_at"])
    except Exception as e:
        print(f"DEBUG REGISTER ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

@router.post("/login")
async def login(user_in: UserLogin):
    user = await db.db["users"].find_one({"email": user_in.email})
    if not user or not verify_password(user_in.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user["id"]})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user["id"]}
