from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialAssistanceChatbotCore:
    def __init__(self, model_name):
        self.model_name = model_name
        self.chat_model = self.initialize_model()
        self.parser = StrOutputParser()

    def initialize_model(self):
        """Initialize the AI model using LangChain (model)"""
        return ChatOpenAI(model=self.model_name)

    def generate_response(self, prompt):
        """Generate the response from the model based on user input"""
        messages = [
            SystemMessage(content="You are an expert in financial assistance, skilled in explaining things in a simple and easy to understand manner."),
            HumanMessage(content=prompt),
        ]
        return self.chat_model.invoke(messages)

    def format_response(self, response):
        """Extract and format the response"""
        return self.parser.invoke(response)

def initialize():
    """Initialize chat history in session state if it doesn't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def add_message_to_history(role, message):
    """Add a message to the session state chat history."""
    st.session_state.messages.append({"role": role, "content": message})

def display_message(role, message):
    """Display the message in the chat UI."""
    with st.chat_message(role):
        st.write(message)


def main():
    """Main function to run the chatbot application."""
    st.title("Financial Assistance Chatbot")
    st.write("This chatbot is designed to help you with financial assistance questions in a simple and easy-to-understand manner.")
    
    model_name = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o-mini"])
    chatbot_core = FinancialAssistanceChatbotCore(model_name)

    initialize()
    display_chat_history()

    prompt = st.chat_input("Enter your question:")
    if prompt:
        add_message_to_history("user", prompt)
        display_message("user", prompt)

        with st.status("AI model response..."):
            response = chatbot_core.generate_response(prompt)
            formatted_response = chatbot_core.format_response(response)
        
        add_message_to_history("assistant", formatted_response)
        display_message("assistant", formatted_response)

# Main entry point
if __name__ == "__main__":
    main()
