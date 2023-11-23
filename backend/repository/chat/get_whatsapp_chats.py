from typing import List
from models import Chat, get_supabase_db

def get_whatsapp_chats(phone: str) -> List[Chat]:
    supabase_db = get_supabase_db()
    response = supabase_db.get_whatsapp_chats(phone)
    chats = [Chat(chat_dict) for chat_dict in response.data]
    return chats