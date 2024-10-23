from sqlalchemy import create_engine, text
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableFieldSpec
import streamlit as st
from langchain_core.output_parsers import StrOutputParser   

# Create an SQLAlchemy engine for the SQLite database
engine = create_engine("sqlite:///memory.db")

# Function to clear data from all tables in the SQLite database
def clear_memory():
    with engine.connect() as connection:
        # Fetch all table names
        result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = result.fetchall()
        
        # Delete data from each table
        for table in tables:
            table_name = table[0]
            connection.execute(text(f"DELETE FROM {table_name}"))
            st.success("Memory has been cleared!")

        connection.commit()

# Define the function to get the session history using SQL database
def get_session_history(user_id: str, session_id: str):
    return SQLChatMessageHistory(session_id=f"{user_id}--{session_id}", connection=engine)

# Set up the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Create the chain by combining the prompt and the model
runnable = prompt | model | StrOutputParser()

# Wrap the runnable with message history
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the session.",
            default="",
            is_shared=True,
        ),
    ],
)

# Streamlit interface
st.title("Chat with Memory")
st.write("This app uses LangChain with SQL-based message history to simulate memory in conversations.")

# Streamlit button to trigger the memory clearing
if st.button("Clear Memory"):
    clear_memory()

# User inputs
user_id = st.text_input("Enter your User ID:")
session_id = st.text_input("Enter your Session ID:")
user_input = st.text_input("Your question:")

if st.button("Send"):
    if user_id and session_id and user_input:
        response = runnable_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"user_id": user_id, "session_id": session_id}},
        )
        if response:
            st.write("AI:", response)

        # Optionally, display the session history
        with st.expander("View Session History"):
            message_history = get_session_history(user_id, session_id)
            for msg in message_history.messages:
                st.write(f"{msg.type.capitalize()}: {msg.content}")
    else:
        st.error("Please fill in all fields.")

