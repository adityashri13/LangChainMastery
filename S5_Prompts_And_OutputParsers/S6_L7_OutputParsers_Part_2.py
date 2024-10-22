from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json
from langchain.globals import set_verbose

set_verbose(True)

# Define the structure of the output data using Pydantic
class LanguageOutputStructure(BaseModel):
    """Information about a sentence."""
    original_sentence: str = Field(description="Sentence asked by user")
    desired_language: str = Field(description="Desired language in which sentence to be translated")
    translated_sentence: str = Field(description="Translated sentence for a given sentence in given language")

# Set up a parser to handle output formatting
parser = PydanticOutputParser(pydantic_object=LanguageOutputStructure)

# Create a prompt template for the system message
system_prompt_template: PromptTemplate = PromptTemplate(
    template="""\n{format_instructions}\n You are a language translator. An English speaker wants to translate
    {original_sentence} to {desired_language}. Tell him the correct answer.""",
    input_variables=["original_sentence", "desired_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Create a prompt template for the human message
user_prompt_template: PromptTemplate = PromptTemplate(
    template="Translate {original_sentence} to {desired_language}",
    input_variables=["original_sentence", "desired_language"],
)

# Wrap the user prompt in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Combine the system and human message templates into a chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_template, user_message_template]
)

# Format the messages by replacing the placeholders with actual values
messages = chat_prompt_template.format_messages(original_sentence="I love Pizza!", desired_language="French")

# Create an instance of the ChatOpenAI model
chat_model: ChatOpenAI = ChatOpenAI()

# Generate the response from the chat model using the messages
chat_model_response = chat_model.invoke(messages)

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the formatted response type
# print(f"AnswerType: {type(formatted_response)}")

# Convert the formatted response to JSON for better readability
json_output = formatted_response.json()
output_dict = json.loads(json_output)

# Print the final formatted response
print(f"Answer: {output_dict}")