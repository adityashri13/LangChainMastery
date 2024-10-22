from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["GOOGLE_API_KEY"]

# Initialize the Anthropic model using LangChain (model)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Initialize the parser to format the output (model)
parser = StrOutputParser()

while True:
    # Take user input
    user_query = input("Enter your question (or type 'quit' to exit): ")

    # Check if the user wants to quit
    if user_query.lower() == 'quit':
        print("Exiting the application. Goodbye!")
        break

    # Prepare the messages to be sent to the model (prompt/input)
    messages = [
        SystemMessage(content="You are an expert in financial assistance, skilled in explaining things in a simple and easy to understand manner."),
        HumanMessage(content=user_query),
    ]

    # Generate the response from the model (model)
    chat_model_response = chat_model.invoke(messages)

    # Extract and format the response (output)
    formatted_response = parser.invoke(chat_model_response)

    # Print the question and the generated response (output)
    print("-------------")
    print(f"Question: {user_query}")
    print("-------------")
    print(f"Raw Answer Type: {type(chat_model_response)}")
    print("-------------")
    print(f"Raw Answer: {chat_model_response}")
    print("-------------")
    print(f"Formatted Answer Type: {type(formatted_response)}")
    print("-------------")
    print(f"Formatted Answer: {formatted_response}")
