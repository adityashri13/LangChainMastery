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
from langchain_core.runnables import RunnableMap, RunnablePassthrough
# from dotenv import load_dotenv

# load_dotenv()
# =====================
# 2. File Setup
# =====================
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# =====================
# 3. Load the Data
# =====================
loader = TextLoader(file_path, encoding='UTF-8')
data_docs = loader.load()

# =====================
# 4. Split the Data into Chunks
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
# 5. Create and Index Embeddings
# =====================
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()