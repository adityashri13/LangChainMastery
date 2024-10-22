from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_core.runnables import RunnableSequence 
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.pydantic_v1 import BaseModel, Field

class PersonOutputParser(BaseModel):
    Achievement_1: str = Field(description="First achievement of a person")
    Achievement_2: str = Field(description="Second achievement of a person")
    Achievement_3: str = Field(description="Third achievement of a person")

# Set up a parser to handle JSON output
parser = PydanticOutputParser(pydantic_object=PersonOutputParser)

# Create a prompt template for the human message
user_prompt_template = PromptTemplate(
    template="What are 3 biggest achievement of {person}?",
    input_variables=["person"],
)

# Wrap the user prompt template in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Create a prompt template for the system message
system_prompt_template = PromptTemplate(
    template="""\n{format_instructions}\nYou are a history guide.
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Wrap the system prompt template in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Combine the system and human messages into a chat prompt template
chat_prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        user_message_template
    ]
)

# template = """
# What is the city {person} is from? 
# Only respond with the name of the city.
# """

# prompt1 = ChatPromptTemplate.from_template(template)

# # prompts
# prompt1 = PromptTemplate.from_template(
#   'What is the city {person} is from? Only respond with the name of the city.'
# )
# prompt2 = PromptTemplate.from_template(
#   'What country is the city {city} in? Respond in {language}.'
# )

# template = """
# What country is the city {city} in? 
# Translate the description of the landmark '{landmark}' into {language}.
# """

# template = """
# What country is the city {city} in? 
# Respond in {language}.
# """

# Create a prompt template for the human message
user_prompt_template = PromptTemplate(
    template="""Write a instagram post about {person}
    on his following achievements:
    {Achievement_1},
    {Achievement_2},
    {Achievement_3}
    in {language} language.""",
    input_variables=["person", "Achievement_1", "Achievement_2", "Achievement_3", "language"],
)

# Wrap the user prompt template in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Create a prompt template for the system message
system_prompt_template = PromptTemplate(
    template="""You are a content writer expert in writing about historical fact in interesting and catchy manner.
    """,
)

# Wrap the system prompt template in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Combine the system and human messages into a chat prompt template
chat_prompt_template_2 = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        user_message_template
    ]
)

# chat_prompt_template_2 = ChatPromptTemplate.from_template(template)

# model
model = ChatOpenAI()

# output parser
output_parser = StrOutputParser()

# chain
# chain = prompt1.pipe(model).pipe(output_parser) # This syntax also works
chain = chat_prompt_template_1 | model | parser
pprint(chain)

# combined chain
combined_chain = RunnableSequence(
    {
        "Achievement_1": chain,
        "Achievement_2": chain,
        "Achievement_3": chain,
        "person": lambda inputs: inputs['person'],
        "language": lambda inputs: inputs['language'],
        # "language": RunnablePassthrough(),
    },
    chat_prompt_template_2,
    model,
    output_parser
)
print("--------")
print("--------")
pprint(combined_chain)

result = combined_chain.invoke({
  "person": "Obama",
  "language": "English",
})
print(result)