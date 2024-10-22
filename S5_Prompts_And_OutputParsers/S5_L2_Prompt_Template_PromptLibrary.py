# prompt_library.py

from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

# Define a system message prompt template for translation tasks
sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["original_sentence", "desired_language"],
    template="""You are a language translator. An English speaker wants to translate
    {original_sentence} to {desired_language}. Tell him the correct answer."""
)
system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

# Define a human message prompt template for translation tasks
user_prompt: PromptTemplate = PromptTemplate(
    input_variables=["original_sentence", "desired_language"],
    template="Translate {original_sentence} to {desired_language}"
)
user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)

# Function to create a chat prompt template for translation tasks
def get_translation_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [system_message_prompt, user_message_prompt]
)