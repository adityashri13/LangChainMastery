import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.prompts import (ChatPromptTemplate, 
 FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data'
file_name = 'AI_Intro.pdf'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# Print the file path to ensure it's correct
print("File Path:", file_path)

# Load the PDF document using PyPDFLoader
loader = PyPDFLoader(file_path)
data_docs = loader.load()

# Split the document into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(data_docs)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Use Chroma as the vector store to index the split documents
vector_store = Chroma.from_documents(split_docs, embeddings)

# Define a retriever to search the vector store
retriever = vector_store.as_retriever()

# Few Shot Examples
examples = [
    {
        "input": "When was the Eiffel Tower built?",
        "output": "What is the history of the Eiffel Tower?",
    },
    {
        "input": "What’s the population of New York City in 2024?",
        "output": "How has New York City's population changed over time?",
    },
]

# Create example prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{input}"),
        SystemMessagePromptTemplate.from_template("{output}")
    ]
)

# Create few-shot examples template
few_shot_example_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"]
)

# Main prompt template for step-back prompting
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a world knowledge expert. Your task is to take a step back and rephrase the question into a broader, more general form that is simpler to answer. Below are some examples:"""
        ),
        few_shot_example_template,
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

# Generate step-back query chain
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
user_query = "How is natural language processing used in voice assistants like Siri and Alexa?"
print(generate_queries_step_back.invoke({"input": user_query}))

# Response prompt for final answer generation
response_prompt_template = """
You are a world knowledge expert. I will ask you a question, and your response should be thorough and consistent with the provided context if it applies. If the context isn't relevant, feel free to disregard it.

# {normal_context}
# {step_back_context}

# Original Question: {input}
# Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# Chain definition
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": itemgetter("input") | retriever,
        # Retrieve context using the step-back question
        "step_back_context": itemgetter("input") | generate_queries_step_back | retriever,
        # Pass on the question
        "input": itemgetter("input"),
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

# Define user query and invoke the chain
result = chain.invoke({"input": user_query})

# Print the result
print(result)
