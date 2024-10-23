from sqlalchemy import create_engine
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser   


# Create an SQLAlchemy engine for the SQLite database
engine = create_engine("sqlite:///memory.db")

# Define the function to get the session history using SQL database
def get_session_history(user_id: str, session_id: str):
    return SQLChatMessageHistory(session_id=f"{user_id}--{session_id}", connection=engine)

# Set up the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Create the chain by combining the prompt and the model
runnable = prompt | model | StrOutputParser()

# Wrap the runnable with message history
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
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

# Interactive loop to continuously get input from the user
while True:
    # Get session_id and user input
    session_id = input("Enter your Session ID (or type 'exit' to quit): ").strip()
    if session_id.lower() == 'exit':
        break

    # Get user_id, conversation_id, and user input
    user_id = input("Enter your User ID (or type 'exit' to quit): ").strip()
    if user_id.lower() == 'exit':
        break

    user_query = input("Ask anything (or type 'exit' to quit): ").strip()
    if user_query.lower() == 'exit':
        break

    # Invoke the chain with the user input and session info
    response = runnable_with_history.invoke(
        {"question": user_query},
        config={"configurable": {"user_id": user_id, "session_id": session_id}},
    )

    print("AI:", response)

