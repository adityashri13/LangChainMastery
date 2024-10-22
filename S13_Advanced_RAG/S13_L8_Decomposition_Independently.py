import os
import pprint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.load import dumps, loads
from langchain import hub

# =====================
# 1. File Setup
# =====================
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# =====================
# 2. Load the Data
# =====================
# Load the data from the file into the `TextLoader`
loader = TextLoader(file_path)
data_docs = loader.load()

# =====================
# 3. Split the Data into Chunks
# =====================
# Use a RecursiveCharacterTextSplitter to split the document into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# Split the documents into chunks
splits = text_splitter.split_documents(data_docs)

# =====================
# 4. Create and Index Embeddings
# =====================
# Use OpenAI embeddings to create vector representations of the document chunks
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

# Set up the retriever from the vectorstore
retriever = vectorstore.as_retriever()

# =====================
# 5. Query Decomposition and Generation
# =====================
# Define the prompt for query decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# Set up a chain for query decomposition
generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Decompose the user's query into sub-questions
user_query = "what do you know of Panagiotis Soutsos"
questions = generate_queries_decomposition.invoke({"question": user_query})
pprint.pprint(questions)

# =====================
# 6. Retrieve and RAG on Sub-Questions
# =====================

# Define a function to retrieve documents and run the RAG process on each sub-question
def retrieve_and_rag(question, sub_question_generator_chain):
    """Retrieve documents and run RAG on each sub-question"""
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.invoke(sub_question)
        answer = (
            ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise."""
            )
            | llm 
            | StrOutputParser()
        ).invoke({"context": retrieved_docs, "question": sub_question})
        rag_results.append(answer)
    return rag_results, sub_questions

# Run the retrieval and RAG process on the sub-questions
answers, questions = retrieve_and_rag(user_query, generate_queries_decomposition)

# =====================
# 7. Format Q&A Pairs
# =====================
# Define a function to format the sub-questions and answers into a readable format
def format_qa_pairs(questions, answers):
    """Format sub-questions and their corresponding answers"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# Format the context for the final RAG chain
context = format_qa_pairs(questions, answers)

# =====================
# 8. Final RAG Answer Generation
# =====================
# Define the final RAG prompt template
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the final RAG chain
final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# Generate the final response by synthesizing an answer from the context
response = final_rag_chain.invoke({"context": context, "question": user_query})
print(response)
