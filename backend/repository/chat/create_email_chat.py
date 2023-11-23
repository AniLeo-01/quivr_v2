from dataclasses import dataclass
from uuid import UUID
from logger import get_logger
from models import Chat, get_supabase_db

logger = get_logger(__name__)

def create_email_chat(chat_id: UUID, email: str) -> Chat:
    supabase_db = get_supabase_db()

    #Chat is created upon the user's first question asked 
    logger.info(f"New chat entry in chats table for user with email {email}")

    #Insert a new row into the chats table
    new_chat = {
        "chat_id": str(chat_id),
        "email": email
    }
    insert_response = supabase_db.create_email_chat(new_chat)
    logger.info(f"Insert response {insert_response.data}")

    return insert_response.data[0]