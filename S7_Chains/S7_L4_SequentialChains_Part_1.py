from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
import json


# Define the OpenAI model to be used
model = ChatOpenAI(model="gpt-4o-mini")

# --------------------------------------------
# --------- Content Chain -------------------
# --------------------------------------------
# Define the prompt template for generating content based on a topic
content_prompt_template = PromptTemplate(
    template="""
    Topic: {topic}
    Content: This is a content for the above topic:
    """,
    input_variables=["topic"]
)

# Wrap the content prompt template in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=content_prompt_template)

# Create a ChatPromptTemplate that includes the system message and the user message template
content_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a content creator. Given the topic, it is your job to write short Instagram reel content, which can be spoken in 30 seconds."),
        user_message_template
    ]
)

# Define the content chain that generates content and formats it
content_chain = content_chat_prompt_template | model | StrOutputParser()

# --------------------------------------------
# --------- Hashtag Chain -------------------
# --------------------------------------------

# Define the prompt template for generating hashtags based on content
hashtag_prompt_template = PromptTemplate(
    template="""
    Content:
    {content}
    Generate Hash tags for the above content:
    """,
    input_variables=["content"]
)

# Wrap the hashtag prompt template in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=hashtag_prompt_template)

# Define the structure of the output data using Pydantic
class ContentOutputStructure(BaseModel):
    """Information about generated content and hashtags."""
    content: str = Field(description="Generated content")
    hashtags: str = Field(description="Generated hashtags")

# Set up a parser to handle output formatting
parser = PydanticOutputParser(pydantic_object=ContentOutputStructure)

# Create a prompt template for the system message with format instructions
system_prompt_template = PromptTemplate(
    template="""\n{format_instructions}\n You are a content creator. Given the content, it is your job to write hashtags for Instagram reels.""",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Create a ChatPromptTemplate that includes the system message and the user message template for hashtags
hashtag_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        user_message_template
    ]
)

# Define the hashtags chain that generates hashtags and formats them
hashtags_chain = hashtag_chat_prompt_template | model | parser

# --------------------------------------------
# --------- Overall Chain -------------------
# --------------------------------------------

# Combine the content chain and hashtags chain in a RunnableSequence
overall_chain = content_chain | hashtags_chain

# Run the overall chain with a given topic
response = overall_chain.invoke({"topic": "Greatest snipers of all time"})

# Convert the formatted response to JSON for better readability
json_output = response.json()
output_dict = json.loads(json_output)

# Print the final formatted response
print(f"JSON Response: {output_dict}")