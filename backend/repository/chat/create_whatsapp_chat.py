from uuid import UUID
from logger import get_logger
from models import Chat, get_supabase_db

logger = get_logger(__name__)

def create_whatsapp_chat(chat_id: UUID, phone: str) -> Chat:
    #creating an instance of the supabase db
    supabase_db = get_supabase_db()
    #logging into the console for debugging
    logger.info(f"New chat entry in chats table for user with phone {phone}")

    #creating an object of the new chat entry
    new_chat = {
        "chat_id": str(chat_id),
        "phone": phone
    }
    insert_response = supabase_db.create_whatsapp_chat(new_chat)
    logger.info(f"Insert response {insert_response.data}")

    return insert_response.data[0]