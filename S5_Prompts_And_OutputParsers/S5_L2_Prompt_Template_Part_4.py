# main.py

from S5_L2_Prompt_Template_PromptLibrary import get_translation_chat_prompt
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -------INPUT
# Get the chat prompt template from the prompt library
chat_prompt:ChatPromptTemplate = get_translation_chat_prompt()

# Format the messages by replacing the placeholders with actual values
formatted_messages = chat_prompt.format_messages(original_sentence="I love Pizza!", desired_language="French")

# Print the formatted messages' content only
print('Formatted messages content:')
for msg in formatted_messages:
    print(msg.content)

# -------MODEL
# Create an instance of the ChatOpenAI model
chat_model: ChatOpenAI = ChatOpenAI()

# Generate the response from the chat model using the formatted messages
chat_model_response = chat_model.invoke(formatted_messages)

# -------OUTPUT
# Initialize a string output parser to process the model's response
parser = StrOutputParser()

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the generated response
print(f"Answer: {formatted_response}")
