from langchain.prompts.prompt import PromptTemplate

_template = """
{prompt}

Context:
{context}

Chat History:
{chat_history}
"""

CHAT_MEMORY_PROMPT = PromptTemplate.from_template(_template)