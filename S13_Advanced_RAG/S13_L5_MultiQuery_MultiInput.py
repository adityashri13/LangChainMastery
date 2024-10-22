import os
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.load import dumps, loads
from dotenv import load_dotenv

load_dotenv()
# =====================
# 1. Setup: Paths and Loading Data
# =====================
script_directory = os.path.dirname(os.path.abspath(__file__))
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'
file_path = os.path.join(script_directory, data_folder, file_name)

loader = TextLoader(file_path)
data_docs = loader.load()

# =====================
# 2. Split the Data into Chunks
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       
    chunk_overlap=20,     
    length_function=len,  
    is_separator_regex=False  
)

splits = text_splitter.split_documents(data_docs)

# =====================
# 3. Create and Index Embeddings
# =====================
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# =====================
# 4. Generate Multiple Query Perspectives (with Multiple Inputs)
# =====================
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question and incorporate the additional context. 
Your goal is to retrieve relevant documents from a vector database by generating 
multiple perspectives on the user question. Provide these alternative questions 
separated by newlines. 

Original question: {question}
Additional context: {additional_context}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# =====================
# 5. Retrieve Relevant Documents (with Multiple Inputs)
# =====================
def get_unique_union(documents):
    """Unique union of retrieved docs."""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

retrieval_chain = generate_queries | retriever.map() | get_unique_union

# =====================
# 6. Create the Final RAG Chain (with Multiple Inputs)
# =====================
template = """Generate a summary based on the following context and user question:

{context}

User Question: {question}
Additional Context: {additional_context}"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {
        "context": retrieval_chain, 
        "question": itemgetter("question"),
        "additional_context": itemgetter("additional_context")
    } 
    | prompt
    | llm
    | StrOutputParser()
)

# =====================
# 7. Execute the RAG Chain with Multiple Inputs
# =====================
input_data = {
    "question": "What do you know of Panagiotis Soutsos?",
    "additional_context": "Focus on his contributions to modern Greek literature."
}

response = final_rag_chain.invoke(input_data)

# Print the final response
print(response)
