
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback


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
model_name = "gpt-4o-mini-0125"
chat_model = ChatOpenAI(model=model_name)

# Generate the response from the model (model)
with get_openai_callback() as cb:
    chat_model_response = chat_model.invoke(messages)
    print('cost analysis:')
    print(cb)
    print(f"Cost calculated by LangChain for using {cb.total_cost}")


# -------OUTPUT
# Initialize the parser to format the output (model)
parser = StrOutputParser()

# Extract and format the response (output)
formatted_response = parser.invoke(chat_model_response)

# Print the question and the generated response (output)
print(f"Question: {user_query}")
print(f"Raw Answer Type: {type(chat_model_response)}")
print(f"Raw Answer: {chat_model_response}")
print(f"Formatted Answer Type: {type(formatted_response)}")
print(f"Formatted Answer: {formatted_response}")