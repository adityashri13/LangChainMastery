from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from a .env file (this should contain your OPENAI_API_KEY)
load_dotenv()

# Create an SQLAlchemy engine for the SQLite database
engine = create_engine("sqlite:///memory_2.db")

# Define the function to get the session history using SQL database
def get_session_history(client_id: str, event_id: str):
    session_id = f"{client_id}--{event_id}"
    return SQLChatMessageHistory(session_id=session_id, connection=engine)

# Set up the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt template for the event planning assistant
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an event planning assistant. Help the user plan their event by remembering all the details they provide."),
        MessagesPlaceholder(variable_name="history"),  # Placeholder for conversation history
        ("human", "{input}"),
    ]
)

# Create the chain by combining the prompt, model, and output parser
runnable = prompt | model | StrOutputParser()

# Wrap the chain with RunnableWithMessageHistory for automatic history management
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="client_id",
            annotation=str,
            name="Client ID",
            description="Unique identifier for the client.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="event_id",
            annotation=str,
            name="Event ID",
            description="Unique identifier for the event.",
            default="",
            is_shared=True,
        ),
    ],
)

# Interactive loop to continuously get input from the user
while True:
    # Get client_id, event_id, and user input
    client_id = input("Enter Client ID (or type 'exit' to quit): ").strip()
    if client_id.lower() == 'exit':
        break
    
    event_id = input("Enter Event ID (or type 'exit' to quit): ").strip()
    if event_id.lower() == 'exit':
        break

    user_input = input("Ask anything (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break

    # Invoke the chain with the user input and session info
    response = runnable_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"client_id": client_id, "event_id": event_id}},
    )

    # Print the AI's response if available
    if response:
        print("AI:", response)
