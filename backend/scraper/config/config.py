from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    supabase_url: str
    supabase_service_key: str
    pg_database_url: str = "not implemented"
    resend_api_key: str = "null"
    resend_email_address: str = "brain@mail.quivr.app"
    CSE_API_KEY: str
    ENGINE_ID: str

def get_settings():
    settings = Settings()
    return settings