from typing import List, Optional
from uuid import UUID
from venv import logger
import os
import re
from bs4 import BeautifulSoup
from logger import get_logger
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Form
from starlette.responses import Response
from fastapi.responses import StreamingResponse
from llm.qa_base import QABaseBrainPicking
from llm.qa_headless import HeadlessQA
from middlewares.auth import AuthBearer, get_current_user
from models import Brain, BrainEntity, Chat, ChatQuestion, UserUsage, get_supabase_db
from models.databases.supabase.chats import QuestionAndAnswer
from modules.user.entity.user_identity import UserIdentity
from modules.user.repository import get_user_identity
from repository.brain import get_brain_details, get_default_user_brain_or_create_new, get_brain_by_phone
from repository.chat import (
    ChatUpdatableProperties,
    CreateChatProperties,
    GetChatHistoryOutput,
    create_chat,
    get_chat_by_id,
    get_user_chats,
    update_chat,
    create_whatsapp_chat,
    get_whatsapp_chats
)
from repository.chat.add_question_and_answer import add_question_and_answer
from repository.chat.get_chat_history_with_notifications import (
    ChatItem,
    get_chat_history_with_notifications,
)
from repository.notification.remove_chat_notifications import remove_chat_notifications
from routes.chat.factory import get_chat_strategy
from routes.chat.utils import (
    NullableUUID,
    check_user_requests_limit,
    delete_chat_from_db,
)
from celery_task import process_and_send_message, process_and_send_email, process_and_send_message_sales


chat_router = APIRouter()

logger = get_logger(__name__)

@chat_router.get("/chat/healthz", tags=["Health"])
async def healthz():
    return {"status": "ok"}


# get all chats
@chat_router.get("/chat", dependencies=[Depends(AuthBearer())], tags=["Chat"])
async def get_chats(current_user: UserIdentity = Depends(get_current_user)):
    """
    Retrieve all chats for the current user.

    - `current_user`: The current authenticated user.
    - Returns a list of all chats for the user.

    This endpoint retrieves all the chats associated with the current authenticated user. It returns a list of chat objects
    containing the chat ID and chat name for each chat.
    """
    chats = get_user_chats(str(current_user.id))
    return {"chats": chats}


# delete one chat
@chat_router.delete(
    "/chat/{chat_id}", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def delete_chat(chat_id: UUID):
    """
    Delete a specific chat by chat ID.
    """
    supabase_db = get_supabase_db()
    remove_chat_notifications(chat_id)

    delete_chat_from_db(supabase_db=supabase_db, chat_id=chat_id)
    return {"message": f"{chat_id}  has been deleted."}


# update existing chat metadata
@chat_router.put(
    "/chat/{chat_id}/metadata", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def update_chat_metadata_handler(
    chat_data: ChatUpdatableProperties,
    chat_id: UUID,
    current_user: UserIdentity = Depends(get_current_user),
) -> Chat:
    """
    Update chat attributes
    """

    chat = get_chat_by_id(chat_id)  # pyright: ignore reportPrivateUsage=none
    if str(current_user.id) != chat.user_id:
        raise HTTPException(
            status_code=403,  # pyright: ignore reportPrivateUsage=none
            detail="You should be the owner of the chat to update it.",  # pyright: ignore reportPrivateUsage=none
        )
    return update_chat(chat_id=chat_id, chat_data=chat_data)


# create new chat
@chat_router.post("/chat", dependencies=[Depends(AuthBearer())], tags=["Chat"])
async def create_chat_handler(
    chat_data: CreateChatProperties,
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    Create a new chat with initial chat messages.
    """

    return create_chat(user_id=current_user.id, chat_data=chat_data)


# add new question to chat
@chat_router.post(
    "/chat/{chat_id}/question",
    dependencies=[
        Depends(
            AuthBearer(),
        ),
    ],
    tags=["Chat"],
)
async def create_question_handler(
    request: Request,
    chat_question: ChatQuestion,
    chat_id: UUID,
    brain_id: NullableUUID
    | UUID
    | None = Query(..., description="The ID of the brain"),
    current_user: UserIdentity = Depends(get_current_user),
) -> GetChatHistoryOutput:
    """
    Add a new question to the chat.
    """

    chat_instance = get_chat_strategy(brain_id)

    chat_instance.validate_authorization(user_id=current_user.id, brain_id=brain_id)

    brain = Brain(id=brain_id)
    brain_details: BrainEntity | None = None

    userDailyUsage = UserUsage(
        id=current_user.id,
        email=current_user.email,
    )
    userSettings = userDailyUsage.get_user_settings()
    is_model_ok = (brain_details or chat_question).model in userSettings.get("models", ["gpt-3.5-turbo"])  # type: ignore

    # Retrieve chat model (temperature, max_tokens, model)
    if (
        not chat_question.model
        or not chat_question.temperature
        or not chat_question.max_tokens
    ):
        # TODO: create ChatConfig class (pick config from brain or user or chat) and use it here
        chat_question.model = chat_question.model or brain.model or "gpt-3.5-turbo"
        chat_question.temperature = (
            chat_question.temperature or brain.temperature or 0.1
        )
        chat_question.max_tokens = chat_question.max_tokens or brain.max_tokens or 512

    try:
        check_user_requests_limit(current_user)
        is_model_ok = (brain_details or chat_question).model in userSettings.get("models", ["gpt-3.5-turbo"])  # type: ignore
        gpt_answer_generator = chat_instance.get_answer_generator(
            chat_id=str(chat_id),
            model=chat_question.model if is_model_ok else "gpt-3.5-turbo",  # type: ignore
            max_tokens=chat_question.max_tokens,
            temperature=chat_question.temperature,
            brain_id=str(brain_id),
            streaming=False,
            prompt_id=chat_question.prompt_id,
            user_id=current_user.id,
        )

        chat_answer = gpt_answer_generator.generate_answer(chat_id, chat_question)

        return chat_answer
    except HTTPException as e:
        raise e


# stream new question response from chat
@chat_router.post(
    "/chat/{chat_id}/question/stream",
    dependencies=[
        Depends(
            AuthBearer(),
        ),
    ],
    tags=["Chat"],
)
async def create_stream_question_handler(
    request: Request,
    chat_question: ChatQuestion,
    chat_id: UUID,
    brain_id: NullableUUID
    | UUID
    | None = Query(..., description="The ID of the brain"),
    current_user: UserIdentity = Depends(get_current_user),
) -> StreamingResponse:
    chat_instance = get_chat_strategy(brain_id)
    chat_instance.validate_authorization(user_id=current_user.id, brain_id=brain_id)

    brain = Brain(id=brain_id)
    brain_details: BrainEntity | None = None
    userDailyUsage = UserUsage(
        id=current_user.id,
        email=current_user.email,
    )

    userSettings = userDailyUsage.get_user_settings()

    # Retrieve chat model (temperature, max_tokens, model)
    if (
        not chat_question.model
        or chat_question.temperature is None
        or not chat_question.max_tokens
    ):
        # TODO: create ChatConfig class (pick config from brain or user or chat) and use it here
        chat_question.model = chat_question.model or brain.model or "gpt-3.5-turbo"
        chat_question.temperature = chat_question.temperature or brain.temperature or 0
        chat_question.max_tokens = chat_question.max_tokens or brain.max_tokens or 256

    try:
        logger.info(f"Streaming request for {chat_question.model}")
        check_user_requests_limit(current_user)
        gpt_answer_generator: HeadlessQA | QABaseBrainPicking
        # TODO check if model is in the list of models available for the user

        is_model_ok = (brain_details or chat_question).model in userSettings.get("models", ["gpt-3.5-turbo"])  # type: ignore

        gpt_answer_generator = chat_instance.get_answer_generator(
            chat_id=str(chat_id),
            model=(brain_details or chat_question).model if is_model_ok else "gpt-3.5-turbo",  # type: ignore
            max_tokens=(brain_details or chat_question).max_tokens,  # type: ignore
            temperature=(brain_details or chat_question).temperature,  # type: ignore
            streaming=True,
            prompt_id=chat_question.prompt_id,
            brain_id=str(brain_id),
            user_id=current_user.id,
        )

        return StreamingResponse(
            gpt_answer_generator.generate_stream(chat_id, chat_question),
            media_type="text/event-stream",
        )

    except HTTPException as e:
        raise e


# get chat history
@chat_router.get(
    "/chat/{chat_id}/history", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def get_chat_history_handler(
    chat_id: UUID,
) -> List[ChatItem]:
    # TODO: RBAC with current_user
    return get_chat_history_with_notifications(chat_id)


@chat_router.post(
    "/chat/{chat_id}/question/answer",
    dependencies=[Depends(AuthBearer())],
    tags=["Chat"],
)
async def add_question_and_answer_handler(
    chat_id: UUID,
    question_and_answer: QuestionAndAnswer,
) -> Optional[Chat]:
    """
    Add a new question and anwser to the chat.
    """
    return add_question_and_answer(chat_id, question_and_answer)

@chat_router.post("/chat/whatsapp")
async def read_item(WaId: str = Form(...), Body: str = Form(...), To: str = Form(...)):
    """
    Generate response from whatsapp without requiring authentication
    """
    logger.info(f"Recepient: {WaId}")
    logger.info(f"Admin: {To}")
    try:
        process_and_send_message.delay(WaId, Body, To)
        return {"status": "processing"}
    except HTTPException as e:
        raise e
    
@chat_router.post("/chat/email", tags=["Chat"])
async def create_email_question_handler(
    Admin_Email: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    Subject: Optional[str] = Form(None),
    Body: Optional[str] = Form(None),
        Reply_To: Optional[str] = Form(None),
):
    # Printing received data for demonstration
    if Reply_To is not None:
        logger.info(f"Reply to: {Reply_To}")
        if "airbnb" in Reply_To.lower():
            logger.info("This is a direct reply to airbnb.")
            From = Reply_To

    #printing the received data for demonstration
    logger.info(f"From: {From}")
    logger.info(f"Subject: {Subject}")
    logger.info(f"Body: {Body}")
    extracted_text = None
    ThreadId = None
    if "<div>" in Body.lower():  # This is a simplistic check to detect html
        Body = extract_user_message(Body)
        extracted_text = extract_user_message(Body)
        ThreadId = extract_thread_id(Body)
        logger.info(f"ThreadId: {ThreadId}")
        logger.info(f"Extracted Text: {Body}")
        logger.info(f"Extracted Text: {extracted_text}")
        email_text = extracted_text
    else:
        email_text = Body

    process_and_send_email.delay(Admin_Email, email_text, From, Subject, ThreadId)
    return {"status": "success", "message": "answer"}
    


@chat_router.post("/chat/whatsapp/sales")
async def read_item(WaId: str = Form(...), Body: str = Form(...), To: str = Form(...)):
    """
#     Generate an answer for a question without requiring authentication.
#     """
    logger.info(f"Recepient: {WaId}")
    logger.info(f"Admin: {To}")
    client = WaId
    admin = To
    process_and_send_message_sales.delay(client, admin, Body)
    try:
        return {"status": "processing"}
    except HTTPException as e:
        raise e


def extract_user_message(email_html_content):
    soup = BeautifulSoup(email_html_content, 'html.parser')

    # First attempt: Try extracting based on a div with specific styles
    user_message_div = soup.find('div', style=lambda x: x and 'font-weight:300;font-family:' in x)
    if user_message_div:
        return user_message_div.get_text(strip=True)

    # Second attempt: Try extracting based on a p within a div with specific styles
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            for cell in row.find_all('td'):
                div = cell.find('div', class_='regular')
                if div:
                    p = div.find('p', class_='regular')
                    if p:
                        user_message = p.get_text(strip=True)
                        return user_message
                    
    # Third attempt: Try extracting based on div with attribute dir="ltr"
    div_dir_ltr = soup.find('div', dir='ltr')
    if div_dir_ltr:
        return div_dir_ltr.get_text(strip=True)

    
    # If no message is found
    return None


def extract_thread_id(email_html_content):
    soup = BeautifulSoup(email_html_content, 'html.parser')

    # Find all <a> tags in the email content
    links = soup.find_all('a', href=True)

    for link in links:
        href = link['href']

        # Use a regular expression to extract the thread ID from the URLs
        thread_id_match = re.search(r'/thread/(\d+)', href)
        if thread_id_match:
            return thread_id_match.group(1)
    return None