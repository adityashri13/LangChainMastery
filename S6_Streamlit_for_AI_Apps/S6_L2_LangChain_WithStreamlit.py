import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# Streamlit Sidebar for Model Selection
model_name = st.sidebar.selectbox("Select Model", ["gemini-1.5-pro-001", "gemini-1.5-flash"])

# Initialize the Anthropic model using LangChain (model)
chat_model = ChatGoogleGenerativeAI(model=model_name)

# Initialize the parser to format the output (model)
parser = StrOutputParser()

# Streamlit UI
st.title("Financial Assistance Chatbot")
st.write("This chatbot is designed to help you with financial assistance questions in a simple and easy-to-understand manner.")

# User input
user_query = st.text_input("Enter your question:")

if user_query:
    # Prepare the messages to be sent to the model
    messages = [
        SystemMessage(content="You are an expert in financial assistance, skilled in explaining things in a simple and easy to understand manner."),
        HumanMessage(content=user_query),
    ]

    # Generate the response from the model
    chat_model_response = chat_model.invoke(messages)

    # Extract and format the response
    formatted_response = parser.invoke(chat_model_response)

    # Display the question and response
    st.write("**Question:**", user_query)
    st.write("**Formatted Answer:**", formatted_response)

    # # Optionally display raw response information
    # st.write("**Raw Answer Type:**", type(chat_model_response))
    # st.write("**Raw Answer:**", chat_model_response)
    # st.write("**Formatted Answer Type:**", type(formatted_response))