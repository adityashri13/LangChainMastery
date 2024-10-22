from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# -------INPUT
user_query = "\
Why do we celebrate Independence day in Japan? Tell me 5 interesting points about it?\
"

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant"
            )
        ),
        HumanMessage(
            content=user_query
        ),
    ]
)

messages = chat_template.format_messages()
print(messages)


# -------MODEL
# Initialize the OpenAI model using LangChain (model)
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Generate the response from the model (model)
chat_model_response = chat_model.invoke(messages)


# -------OUTPUT
# Initialize the parser to format the output (model)
parser = StrOutputParser()

# Extract and format the response (output)
formatted_response = parser.invoke(chat_model_response)

# Print the question and the generated response (output)
print(f"Question: {user_query}")
print(f"Answer: {formatted_response} ")

