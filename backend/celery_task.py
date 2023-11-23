from celery import shared_task
from models.brains import Brain
from models.settings import get_supabase_db
from packages.embeddings.vectors import Neurons
from repository.files.upload_file import DocumentSerializable
from twilio.rest import Client
from venv import logger
import os
import re
from modules.user.repository import get_user_identity
from repository.brain import get_default_user_brain_or_create_new, get_brain_by_phone, get_brain_by_email
from llm.qa_base import QABaseBrainPicking
from models import ChatQuestion
from repository.chat import get_whatsapp_chats, create_chat, create_whatsapp_chat, CreateChatProperties, create_email_chat, get_email_chats
from email.mime.multipart import MIMEMultipart
import email.mime.text import MIMEText
import smtplib
from fastapi import HTTPException
import openai
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate
    )


@shared_task
def create_embedding_for_document(brain_id, doc_with_metadata, file_sha1):
    neurons = Neurons()
    doc = DocumentSerializable.from_json(doc_with_metadata)
    created_vector = neurons.create_vector(doc)
    database = get_supabase_db()
    database.set_file_sha_from_metadata(file_sha1)

    created_vector_id = created_vector[0]  # pyright: ignore reportPrivateUsage=none

    brain = Brain(id=brain_id)  # pyright: ignore
    brain.create_brain_vector(created_vector_id, file_sha1)


@shared_task
def process_and_send_message(WaId, Body, To):
    admin_number = To
    #fetch the user_id of the ADMIN from the env file
    user_id = os.environ.get("ADMIN_USER_ID")
    #using the admin user_id, fetch the user identity object
    user = get_user_identity(user_id)
    #fetch brain to get the chats associated with the admin user
    split_text = admin_number.split("whatsapp:")

    # Getting the second part which is the phone number
    match = split_text[1]
    logger.info(f"admin number: {match}")
    brain_phone = match
    brain = get_brain_by_phone(brain_phone)
    logger.info(f"Brain fetched for phone {brain_phone}: {brain}")

    if not brain:
        logger.info('No Brain fetched for phone... Fetching default brain for user')
        #create the brain for the user, if it doesn't exist create a new brain object
        brain = get_default_user_brain_or_create_new(user)
    
    #create a ChatQuestion object with the whatsapp id and the question
    chat_question = ChatQuestion(question=Body, WaId = WaId)
    chat_question.brain_id = brain.id

    #fetch the existing whatsapp chats for the WaID if any
    existing_chat = get_whatsapp_chats(WaId)
    #if its a new chat
    if len(existing_chat) == 0:
        #here we are setting the name of the chat to the WaId (we can define it by the name as well)
        chat = create_chat(user_id=user_id, chat_data=CreateChatProperties(name=WaId))
        chat_id = chat['chat_id']
        create_whatsapp_chat(chat_id=chat_id, phone=WaId)
    #if its an existing chat
    else:
        chat = existing_chat[0]
        chat_id = chat["chat_id"]

    try:
        if brain:
            #create a QABaseBrainPicking object
            gpt_answer_generator = QABaseBrainPicking(
                chat_id=str(chat_id),
                model='gpt-4',
                brain_id=str(brain.id),
                streaming=True,
                temperature=0.1,
            )
            chat_answer = gpt_answer_generator.generate_answer(chat_id, chat_question)
        send_whatsapp_message(chat_answer.assistant, admin_number, WaId)

def send_whatsapp_message(message, from_, to):
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    logger.info(f"admin: {from_} , client: {to}")
    client.messages.create(
        body=message,
        from_=from_,
        to=f'whatsapp:{to}'
    )

@shared_task
def process_and_send_message_sales(client_number,admin_number,question):

    # You can optionally validate the brain_id here if needed.

    # Get user identity from the ADMIN_USER_ID set as env variable
    user_id = os.environ.get("ADMIN_USER_ID")
    user = get_user_identity(user_id)

    # Get brain to associate admin users chat with. If no brain exists
    # with the reciever phone get default user brain or create new brain

    
    logger.info(f"admin number: {admin_number}")

    brain = get_brain_by_phone(client_number)
    logger.info(f'Brain fetched by client phone: {client_number}:  {brain}')

    if not brain:
        logger.info(f"No Brain found for client phone {client_number}")
        message = "Thank you for reaching out to us. We are currently in the process of configuring your personalized chatbot to ensure an optimized experience. We appreciate your patience and encourage you to check back later for access to your customized chat service."
        send_whatsapp_message(message, admin_number, client_number)
        return

    # Create Chat Question and set its brain_id
    chat_question = ChatQuestion(question=question, WaId=client_number)
    chat_question.brain_id = brain.id

    # Check for existing chats with the same client phone and create
    # new chat if no previous chatb exists
    existing_chat = get_whatsapp_chats(client_number)
    if len(existing_chat) == 0:
        chat = create_chat(user_id=user_id, chat_data=CreateChatProperties(name=client_number))
        chat_id = chat['chat_id']
        create_whatsapp_chat(chat_id=chat_id, phone=client_number)
    else:
        chat = existing_chat[0]
        chat_id = chat.chat_id

    if brain:
        gpt_answer_generator = QABaseBrainPicking(
            chat_id=str(chat_id),
            model="gpt-4",  # type: ignore
            brain_id=str(brain.id),
            streaming=True,
            temperature=0.1,
            max_tokens=256,
            prompt_id=None
        )
        # Assuming `generate_answer` is a function that generates the answer.
        chat_answer = gpt_answer_generator.generate_answer(chat_id, chat_question)

    send_whatsapp_message(chat_answer.assistant, admin_number, client_number)




@shared_task
def send_alert(ai_response,human_input):
    chat = ChatOpenAI(model='gpt-4')
    template="""
    Situation: An interaction is taking place between a customer and a support agent concerning a specific inquiry or issue.

    Conversation Analysis:
    Customer: {human_input}
    Support Agent: {ai_response}

    Assessment Criteria: Review the support agent's latest response to decide if the customer's issue is moving towards a resolution or if additional steps are necessary. The outcome should be one of the following:

    1. 'Resolved' if the support agent's response adequately addresses the customer's concern, and there are no indications of further action needed, also for normal chat.

    2. 'Manager assistance required' if:
    - The response from the agent mentions or implies a need to involve a manager.
    - The customer expresses dissatisfaction or anger that seems to warrant managerial attention.
    - The issue presented is complex or unusual, suggesting that it may be outside the agent's purview and require a manager's input.

    Determination: Resolved/Manager assistance required
    Helpful Answer:
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    # get a chat completion from the formatted messages
    output = chat(chat_prompt.format_prompt(
        human_input=human_input,
        ai_response=ai_response
    ).to_messages())
    output = output.content
    
    logger.info(output)

    if not 'resolved' in output.lower():
        message_body = (
            f"Alert: A customer issue needs further assistance!\n"
            f"Question: {human_input}\n"
            f"AI Response: {ai_response}\n"
            f"Please review and take the necessary action."
        )
        from_ = 'whatsapp:+19544660669'
        to = f'{os.environ.get("ADMIN_NUMBER")}'
        send_whatsapp_message(message_body, from_, to)


@shared_task
def process_and_send_email(Admin_Email, Body, From, Subject, ThreadId=None):
    if ThreadId == None:
        ThreadId = From
    #get user identity from the ADMIN_USER_ID set as env variable
    user_id = os.environ.get("ADMIN_USER_ID")
    user = get_user_identity(user_id)

    #here you might want to fetch or create a 'brain' or context for the email conversation
    brain = get_brain_by_email(Admin_Email)
    
    logger.info(f'Brain fetched for email {Admin_Email}: {brain}')

    if not brain:
        logger.info("No brain found for receiver email...Fetching default brain for the user")
        brain = get_default_user_brain_or_create_new(user)

    #create email question and set it's brain_id or context
    email_question = ChatQuestion(question=Body)
    email_question.brain_id = brain.id

    #check for existing email threads with the same EmailId and create new if no previous exists
    existing_chat = get_email_chats(ThreadId)
    if len(existing_chat) == 0:
        chat = create_chat(user_id=user_id, chat_data=CreateChatProperties(name=ThreadId))
        chat_id = chat['chat_id']
        create_email_chat(chat_id=chat_id, email=ThreadId)
    else:
        chat = existing_chat[0]
        chat_id = chat.chat_id

    if brain:
        gpt_answer_generator = QABaseBrainPicking(
            chat_id=str(chat_id),
            model='gpt-4',
            brain_id=str(brain.id),
            streaming=True,
            temperature=0.1,
        )
        email_answer = gpt_answer_generator.generate_answer(chat_id, email_question)

    logger.info(email_answer)

    send_email(From, email_answer.assistant, Subject, brain)

def send_email(recepient_email: str, email_answer: str, Subject, brain):
    logger.info(f'recipient_email: {recepient_email}')
    try:
        # email_subject = f"re: {Subject}"
        # if "marco" in recepient_email.lower() and "airbnb" in recepient_email.lower():
        #     logger.info("The reply-to address includes 'marco'.")
        #     sender_email = "info@seataya.com"  # Replace with your Gmail email
        #     sender_password = "zcrg pzpn vufm rsup"  # Replace with your Gmail password
        #     sender_name = "Seataya"  # Replace with your name or alias
        # else:
        #     sender_email = "info@gptlab.dev"  # Replace with your Gmail email
        #     sender_password = "abdj robl ltps ogap"  # Replace with your Gmail password
        #     sender_name = "gptlab"  # Replace with your name or alias

        # making sure to keep the subject same for marco airbnb
        if "airbnb" not in recepient_email.lower():
            Subject = f"re: {Subject}"

        sender_email = brain.email
        sender_password = brain.email_password
        sender_name = brain.name
        logger.info(f'sender_email: {sender_email}')
        logger.info(f'sender_password: {sender_password}')
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        msg = MIMEMultipart()
        msg['From'] = f"{sender_name} <{sender_email}>"
        msg['To'] = recepient_email
        msg['Subject'] = Subject

        html = f"""<p>{email_answer}</p> """
        msg.attach(MIMEText(html, 'html'))

        text = msg.as_string()
        server.sendmail(sender_email, recepient_email, text)

        server.quit()
        print("Email sent successfully!")  # Using print statement for simplicity
    except Exception as e:
        print(f"An error occurred: {e} {recepient_email}")  # Using print statement for simplicity
