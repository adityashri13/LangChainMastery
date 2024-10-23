import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# Load the document using TextLoader
loader = TextLoader(file_path,encoding='UTF-8')
data_docs = loader.load()

with open("data_docs.txt", "w") as output:
    output.write(str(data_docs))


# Initialize the RecursiveCharacterTextSplitter to split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# # Split the loaded document into chunks
chunks = text_splitter.split_documents(data_docs)
# pprint.pprint(chunks)  # Uncomment to print the chunks

with open("chunks_100_20.txt", "w") as output:
    output.write(str(chunks))

with open("chunks_100_20.pkl", "wb") as file:
    pickle.dump(chunks, file)

# with 500 and 50
# Initialize the RecursiveCharacterTextSplitter to split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Set a small chunk size for demonstration
    chunk_overlap=50,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# # Split the loaded document into chunks
chunks = text_splitter.split_documents(data_docs)
# pprint.pprint(chunks)  # Uncomment to print the chunks

with open("chunks_500_50.txt", "w") as output:
    output.write(str(chunks))

with open("chunks_500_50.pkl", "wb") as file:
    pickle.dump(chunks, file)


# retrieve with 100
with open("chunks_100_20.pkl", "rb") as file:
    chunks = pickle.load(file)

# retrieve with 500
with open("chunks_500_50.pkl", "rb") as file:
    chunks = pickle.load(file)