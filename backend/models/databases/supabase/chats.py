from typing import Optional
from uuid import UUID

from models.chat import Chat
from models.databases.repository import Repository
from pydantic import BaseModel


class CreateChatHistory(BaseModel):
    chat_id: UUID
    user_message: str
    assistant: str
    prompt_id: Optional[UUID]
    brain_id: Optional[UUID]


class QuestionAndAnswer(BaseModel):
    question: str
    answer: str


class Chats(Repository):
    def __init__(self, supabase_client):
        self.db = supabase_client

    def create_chat(self, new_chat):
        response = self.db.table("chats").insert(new_chat).execute()
        return response
    
    #create a new whatsapp chat in the chat_whatsapp table
    def create_whatsapp_chat(self, new_chat):
        response = self.db.table("chat_whatsapp").insert(new_chat).execute()
        return response
    
    #create a new email chat in the chat_email table
    def create_email_chat(self, new_chat):
        response = self.db.table("chat_email").insert(new_chat).execute()
        return response

    def get_chat_by_id(self, chat_id: str):
        response = (
            self.db.from_("chats")
            .select("*")
            .filter("chat_id", "eq", chat_id)
            .execute()
        )
        return response

    def add_question_and_answer(
        self, chat_id: UUID, question_and_answer: QuestionAndAnswer
    ) -> Optional[Chat]:
        response = (
            self.db.table("chat_history")
            .insert(
                {
                    "chat_id": str(chat_id),
                    "user_message": question_and_answer.question,
                    "assistant": question_and_answer.answer,
                }
            )
            .execute()
        ).data
        if len(response) > 0:
            response = Chat(response[0])

        return None

    def get_chat_history(self, chat_id: str):
        reponse = (
            self.db.from_("chat_history")
            .select("*")
            .filter("chat_id", "eq", chat_id)
            .order("message_time", desc=False)  # Add the ORDER BY clause
            .execute()
        )

        return reponse

    def get_user_chats(self, user_id: str):
        response = (
            self.db.from_("chats")
            .select("chat_id,user_id,creation_time,chat_name")
            .filter("user_id", "eq", user_id)
            .order("creation_time", desc=False)
            .execute()
        )
        return response

    #fetch whatsapp chats from the chat_whatsapp table with phone as filter and order by created_at
    def get_whatsapp_chats(self, phone: str):
        response = (
            self.db.from_("chat_whatsapp")
            .select("chat_id,phone,created_at")
            .filter("phone", "eq", phone)
            .order("created_at", desc=False)
            .execute()
        )
        return response

    def get_email_chats(self, email: str):
        response = (
            self.db.from_("chat_email")
            .select("chat_id,email,created_at")
            .filter("email", "eq", email)
            .order("created_at", desc=False)
            .execute()
        )
        return response
    
    def update_chat_history(self, chat_history: CreateChatHistory):
        response = (
            self.db.table("chat_history")
            .insert(
                {
                    "chat_id": str(chat_history.chat_id),
                    "user_message": chat_history.user_message,
                    "assistant": chat_history.assistant,
                    "prompt_id": str(chat_history.prompt_id)
                    if chat_history.prompt_id
                    else None,
                    "brain_id": str(chat_history.brain_id)
                    if chat_history.brain_id
                    else None,
                }
            )
            .execute()
        )

        return response

    def update_chat(self, chat_id, updates):
        response = (
            self.db.table("chats").update(updates).match({"chat_id": chat_id}).execute()
        )

        return response

    def update_message_by_id(self, message_id, updates):
        response = (
            self.db.table("chat_history")
            .update(updates)
            .match({"message_id": message_id})
            .execute()
        )

        return response

    def get_chat_details(self, chat_id):
        response = (
            self.db.from_("chats")
            .select("*")
            .filter("chat_id", "eq", chat_id)
            .execute()
        )
        return response

    def delete_chat(self, chat_id):
        self.db.table("chats").delete().match({"chat_id": chat_id}).execute()

    def delete_chat_history(self, chat_id):
        self.db.table("chat_history").delete().match({"chat_id": chat_id}).execute()
