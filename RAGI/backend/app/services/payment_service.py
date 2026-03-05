from app.db.mongodb import db
from datetime import datetime
import uuid

class PaymentService:
    @staticmethod
    async def submit_verification(user_id: str, utr_number: str, email: str, type: str = "upgrade"):
        """Submits a new payment/support verification request."""
        verification_id = str(uuid.uuid4())
        now = datetime.utcnow()
        verification_data = {
            "id": verification_id,
            "user_id": user_id,
            "utr_number": utr_number,
            "email": email,
            "type": type,
            "status": "pending",
            "created_at": now,
            "created_date": now.strftime("%Y-%m-%d"),
            "created_time": now.strftime("%H:%M:%S"),
            "updated_at": now
        }
        await db.db["payment_verifications"].insert_one(verification_data)
        return verification_id

    @staticmethod
    async def get_user_verifications(user_id: str):
        """Retrieves all verification requests for a user."""
        cursor = db.db["payment_verifications"].find({"user_id": user_id}).sort("created_at", -1)
        return await cursor.to_list(length=10)
