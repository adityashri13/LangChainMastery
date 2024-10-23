import os
import pprint
from langchain_community.document_loaders import TextLoader

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
loader = TextLoader(file_path, encoding='UTF-8')
data_docs = loader.load()

# Print the type and content of the loaded data
print(type(data_docs))
pprint.pprint(data_docs)
print("Data Loaded Successfully!")