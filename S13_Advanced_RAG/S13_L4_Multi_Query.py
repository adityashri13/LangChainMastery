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
loader = TextLoader(file_path,encoding='UTF-8')
data_docs = loader.load()

# =====================
# 3. Split the Data into Chunks
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# Make splits
splits = text_splitter.split_documents(data_docs)

# =====================
# 4. Create and Index Embeddings
# =====================
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# =====================
# 5. Generate Multiple Query Perspectives
# =====================
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_options = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_options 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
# user_query = "what do you know of Panagiotis Soutsos"
# print(generate_queries.invoke({"question": user_query}))


# Uncomment to test with sample queries
# 5 Sample Prompts for Marketing Strategy Consulting
# user_query = "How can we grow our brand?"
# user_query = "What are the best strategies to increase customer loyalty?"
# user_query = "How can social media improve brand visibility?"
# user_query = "What are effective digital marketing strategies for small businesses?"
# user_query = "How to increase customer engagement online?"

# print(generate_queries.invoke({"question": user_query}))

# =====================
# 6. Retrieve Relevant Documents
# =====================
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Combine steps into a retrieval chain
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# user_query = "what do you know of Panagiotis Soutsos"
# docs = retrieval_chain.invoke({"question": user_query})
# print(len(docs))
# pprint.pprint(docs)


# =====================
# 7. Create the RAG (Retrieval-Augmented Generation) Chain
# =====================
# Define the final RAG prompt template
template = """Answer the following question based on this context:

{context}

User_Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the final RAG chain
llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {
        "context": retrieval_chain, 
        "question": itemgetter("question")
    } 
    | prompt
    | llm
    | StrOutputParser()
)

# =====================
# 8. Execute the RAG Chain
# =====================
user_query = "what do you know of Panagiotis Soutsos"
response = final_rag_chain.invoke({"question": user_query})

# Print the final response
print(response)