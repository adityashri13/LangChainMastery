# Install the necessary packages using the following commands:
# pip install langchain-openai

import os
from typing import Tuple, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Initialize the OpenAI model with the API key from the environment
llm: ChatOpenAI = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

# Define the prompt template for generating email responses
prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that generates email responses. Here is the email you need to respond to:\n\n"
             "{email_text}\n\nGenerate a {response_length} response. "
             "Tone: {tone}. "
             "Personalization Level: {personalization_level}. "
             "Language and Grammar Complexity: {language_complexity}. "
             "Urgency and Priority: {urgency}. "
             "{additional_instructions}")
])

# Create the chain
chain = prompt | llm | StrOutputParser()

# --------------------------------------------
# --------- Main Execution -------------------
# --------------------------------------------

# Example user inputs
email_text: str = "Dear support, I am facing an issue with my order. The delivery is delayed, and I need it urgently. Can you please help?"

response_length: str = "medium"
# Options: 'short', 'medium', 'long'

tone: str = "Professional"
# Options: 'Formal', 'Semi-Formal', 'Informal', 'Casual', 'Friendly', 'Professional'

personalization_level: str = "High"
# Options: 'High', 'Medium', 'Low'

language_complexity: str = "Intermediate"
# Options: 'Simple', 'Intermediate', 'Advanced'

urgency: str = "High"
# Options: 'High', 'Medium', 'Low'

additional_instructions: str = "Please ensure to acknowledge the urgency and offer a potential solution."
# Optional: Additional instructions to customize the response further

# Generate a response to the provided email text using LangChain
response: str = chain.invoke({
    "email_text": email_text,
    "response_length": response_length,
    "tone": tone,
    "personalization_level": personalization_level,
    "language_complexity": language_complexity,
    "urgency": urgency,
    "additional_instructions": additional_instructions
}).strip()

print("\nGenerated Response:\n", response)