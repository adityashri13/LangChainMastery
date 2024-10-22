# Step 1: Import necessary libraries from LangChain and OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Step 2: Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Language Translator. Answer all questions in Hindi."),
    ("placeholder", "{messages}"),
])

# Step 3: Initialize the model
model = ChatOpenAI()

parser = StrOutputParser()

# Step 4: Create a chain
chain = prompt | model | parser

# Step 5: Initialize in-memory chat history
chat_history = InMemoryChatMessageHistory()

# Start a loop to continuously get input from the user
user_query = input('Ask anything to get Answer in Hindi or type "exit" to end: ')

while user_query.lower() != 'exit':
    # Step 7: Add the user's new input to the conversation history
    chat_history.add_user_message(user_query)

    # Step 8: Invoke the chain with the updated chat history
    response = chain.invoke({
        "messages": chat_history.messages,
    })

    # Step 9: Add the AI's response to the conversation history
    chat_history.add_ai_message(response)  # Ensure to extract 'text' from response

    # Step 10: Output the AI's response
    print(response)

    # Get the next input from the user
    user_query = input('Ask anything to get Answer in Hindi or type "exit" to end: ')