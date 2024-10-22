from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Create a prompt template for the system message
system_prompt_template = PromptTemplate(
    template=" You are a travel expert helping to find efficient and comfortable way of travelling from point A to point B."
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Create a prompt template for the human message
user_prompt_template = PromptTemplate(
    template="How to reach {source} from {target}?",
    input_variables=["source", "target"]
)

# Wrap the user prompt in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Combine the system and human message templates into a chat prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_template, user_message_template]
)

user_input_source = input('Enter start destination=')
user_input_target = input('Enter end destination=')

# Format the messages by replacing the placeholders with actual values
messages = chat_prompt_template.format_messages(source=user_input_source, target=user_input_target)

# Create an instance of the ChatOpenAI model
chat_model = ChatOpenAI()

# Generate the response from the chat model using the messages
chat_model_response = chat_model.invoke(messages)

# Initialize a string output parser to process the model's response
parser = StrOutputParser()

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the generated response
print(f"Answer: {formatted_response}")