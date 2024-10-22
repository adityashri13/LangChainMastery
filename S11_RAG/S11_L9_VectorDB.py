import os
import pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# Print the file path to ensure it's correct
print("File Path:", file_path)

# Load the PDF document using PyPDFLoader
loader = PyPDFLoader(file_path)
data_docs = loader.load()

# Print the type and content of the loaded data
print(type(data_docs))
pprint.pprint(data_docs)
print("Data Loaded Successfully!")

# Initialize the RecursiveCharacterTextSplitter to split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# Split the loaded document into chunks
chunks = text_splitter.split_documents(data_docs)
# pprint.pprint(chunks)  # Uncomment to print the chunks

# Create embeddings for the chunks using OpenAIEmbeddings
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=OpenAIEmbeddings()
)

# The vector store is now created and can be used for retrieval