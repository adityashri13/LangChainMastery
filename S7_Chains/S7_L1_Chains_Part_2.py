from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json

class LanguageOutputStructure(BaseModel):
    """Pydantic model to define the structure of the JSON output from the language model."""
    original_sentence: str = Field(description="Sentence asked by user")
    desired_language: str = Field(description="Desired language in which sentence is to be translated")
    translated_sentence: str = Field(description="Translated sentence in the specified language")

# Set up a parser to handle JSON output
parser = PydanticOutputParser(pydantic_object=LanguageOutputStructure)

# Create a prompt template for the human message
user_prompt_template = PromptTemplate(
    template="Translate {original_sentence} to {desired_language}",
    input_variables=["original_sentence", "desired_language"],
)

# Wrap the user prompt template in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Create a prompt template for the system message
system_prompt_template = PromptTemplate(
    template="""\n{format_instructions}\nYou are a language translator. An English speaker wants to translate
    {original_sentence} to {desired_language}. Provide the translation in JSON format with the key "translated_sentence".""",
    input_variables=["original_sentence", "desired_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Wrap the system prompt template in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Combine the system and human messages into a chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        user_message_template
    ]
)

# Create an instance of the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", verbose=True)

# Create a chain with the chat prompt template, model, and parser
chain = chat_prompt_template | model | parser

# Run the chain with input values and print the response
response = chain.invoke({"original_sentence": "I love Pizza!", "desired_language": "Spanish"})

# Print the formatted response type
# print(f"AnswerType: {type(response)}")

# Convert the formatted response to JSON for better readability
json_output = response.json()
output_dict = json.loads(json_output)

# Print the final formatted response
print(f"Answer: {output_dict}")