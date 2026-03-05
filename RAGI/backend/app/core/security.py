from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
import os
import hashlib
import base64

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key_change_me_in_prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _prepare_password(password: str) -> str:
    """
    Standardizes password length using SHA-256 before hashing.
    This bypasses the 72-character limit of Bcrypt and ensures
    consistency across different OS environments.
    """
    # 1. SHA-256 hash the password
    sha_hash = hashlib.sha256(password.encode('utf-8')).digest()
    # 2. Base64 encode it so it's a valid string for Passlib
    return base64.b64encode(sha_hash).decode('utf-8')

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(_prepare_password(plain_password), hashed_password)

def get_password_hash(password):
    return pwd_context.hash(_prepare_password(password))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
