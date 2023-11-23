from typing import List
from models import Chat, get_supabase_db

def get_email_chats(email: str) -> List[Chat]:
    supabase_db = get_supabase_db()
    response = supabase_db.get_email_chats(email)
    chats = [Chat(chat_dict) for chat_dict in response.data]
    return chats