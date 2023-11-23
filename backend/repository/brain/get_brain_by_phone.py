from uuid import UUID
from models import BrainEntity, get_supabase_db
from typing import Optional

def get_brain_by_phone(phone: str) -> Optional[BrainEntity]:
    supabase_db = get_supabase_db()
    return supabase_db.get_brain_by_phone(phone)