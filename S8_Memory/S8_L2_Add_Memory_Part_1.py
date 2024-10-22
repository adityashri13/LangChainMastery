# Import necessary libraries from LangChain and OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# # Step 1: Create a prompt template
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("You are a Language Translator. Answer all questions in Hindi.")),
        MessagesPlaceholder("history"),
    ]
)

# chat_prompt_template.format_messages() # raises KeyError

# chat_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content=("You are a Language Translator. Answer all questions in Hindi.")),
#         MessagesPlaceholder("history", optional=True),
#     ]
# )
# chat_prompt_template.format_messages() # returns empty list []

# chat_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content=("You are a Language Translator. Answer all questions in Hindi.")),
#         MessagesPlaceholder("history", optional=True),
#         HumanMessage("{question}"),
#     ]
# )

# chat_prompt_template.invoke(
#    {
#        "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
#        "question": "now multiply that by 4"
#    }
# )


# Step 2: Initialize the model
model = ChatOpenAI()

parser = StrOutputParser()

# Step 3: Create a chain
chain = chat_prompt_template | model | parser

# Step 4: Define the initial conversation history
conversation_history = []

# Start a loop to continuously get input from the user
user_query = input('Ask anything to get Answer in Hindi or type "exit" to end: ')

while user_query.lower() != 'exit':
    # Step 5: Add the user's query to the conversation history
    new_message = ("human", user_query)
    conversation_history.append(new_message)

    # Step 6: Invoke the chain with the updated history and get the AI response
    response = chain.invoke({"history": conversation_history})
    
    # Step 7: Add the AI's response to the conversation history
    new_model_message = ("ai", response)  # Ensure you extract 'text' from response
    conversation_history.append(new_model_message)

    # Step 8: Output the AI's response
    print(response)

    # Get the next input from the user
    user_query = input('Ask anything to get Answer in Hindi or type "exit" to end: ')
