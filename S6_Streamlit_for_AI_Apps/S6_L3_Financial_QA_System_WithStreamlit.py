import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # Importing the OpenAI chat model
from dotenv import load_dotenv

load_dotenv()

# Streamlit Sidebar for Model Selection
model_name = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o-mini"])

# Initialize the AI model using LangChain (model)
chat_model = ChatOpenAI(model=model_name)

# Initialize the parser to format the output (model)
parser = StrOutputParser()

# Streamlit UI
st.title("Financial Assistance Chatbot")
st.write("This chatbot is designed to help you with financial assistance questions in a simple and easy-to-understand manner.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI with Streamlit
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept user input with chat_input
prompt = st.chat_input("Enter your question:")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Show status while processing
    with st.status("AI model response..."):
        # Prepare the messages to be sent to the model
        messages = [
            SystemMessage(content="You are an expert in financial assistance, skilled in explaining things in a simple and easy to understand manner."),
            HumanMessage(content=prompt),
        ]

        # Generate the response from the model
        chat_model_response = chat_model.invoke(messages)

        # Extract and format the response
        formatted_response = parser.invoke(chat_model_response)
    
    # Display AI response and add to chat history
    with st.chat_message("assistant"):
        st.write(formatted_response)
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
