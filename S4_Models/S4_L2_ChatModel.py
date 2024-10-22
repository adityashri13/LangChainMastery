from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -------INPUT
# Define the user query (prompt/input)
user_query = "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge"

# Prepare the messages to be sent to the model (prompt/input)
messages = [
    SystemMessage(content="You are a helpful assistant, skilled in explaining things in a simple and easy to understand manner."),
    HumanMessage(content=user_query),
]

# -------MODEL
# Initialize the OpenAI model using LangChain (model)
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Generate the response from the model (model)
chat_model_response = chat_model.invoke(messages)


# -------OUTPUT
# Initialize the parser to format the output (model)
parser = StrOutputParser()

# Extract and format the response (output)
formatted_response = parser.invoke(chat_model_response)

# Print the question and the generated response (output)
print(f"Question: {user_query}")
print("------------")
print(f"Raw Answer Type: {type(chat_model_response)}")
print("------------")
print(f"Raw Answer: {chat_model_response}")
print("------------")
print(f"Formatted Answer Type: {type(formatted_response)}")
print("------------")
print(f"Formatted Answer: {formatted_response}")