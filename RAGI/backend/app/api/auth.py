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
        print(f"STEP 1: Checking existing user for {user_in.email}")
        existing_user = await db.db["users"].find_one({"email": user_in.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_id = str(uuid.uuid4())
        
        print("STEP 2: Hashing password...")
        try:
            hashed_pw = get_password_hash(user_in.password)
        except Exception as hash_err:
            print(f"HASHING ERROR: {str(hash_err)}")
            raise Exception(f"Password hashing failed: {str(hash_err)}")

        print("STEP 3: Preparing user dictionary")
        user_dict = {
            "id": user_id,
            "email": user_in.email,
            "hashed_password": hashed_pw,
            "created_at": datetime.utcnow(),
            "usage": {
                "files_uploaded": 0,
                "api_calls": 0,
                "total_bytes": 0
            }
        }
        
        print("STEP 4: Inserting into MongoDB...")
        try:
            await db.db["users"].insert_one(user_dict)
        except Exception as db_err:
            print(f"DATABASE INSERT ERROR: {str(db_err)}")
            raise Exception(f"DB Insert failed: {str(db_err)}")

        print("STEP 5: Registration successful")
        return UserResponse(id=user_id, email=user_in.email, created_at=user_dict["created_at"])
        
    except HTTPException as he:
        # Re-raise HTTP exceptions (like 400 Already Registered)
        raise he
    except Exception as e:
        print(f"DEBUG REGISTER ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return a JSON with the error instead of just raising it
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": "Registration crash", "error": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

@router.post("/login")
async def login(user_in: UserLogin):
    user = await db.db["users"].find_one({"email": user_in.email})
    if not user or not verify_password(user_in.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user["id"]})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user["id"]}
