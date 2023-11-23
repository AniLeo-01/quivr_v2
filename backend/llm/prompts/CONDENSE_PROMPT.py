from langchain.prompts.prompt import PromptTemplate

_template = """Given the following Input at the end, write the Input value exactly as it is but discard User Input key.

User Input: {question}
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
