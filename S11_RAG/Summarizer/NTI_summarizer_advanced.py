
import logging
import os
from typing import List, Tuple, Dict, Union
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.callbacks import get_openai_callback
import pprint
from config import FileConfig, ModelConfig
from S11_RAG.Budget_Summarizer.NTI_logging_config import setup_logging
from dotenv import load_dotenv

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Use enums for configuration
FILE_NAME = FileConfig.FILE_NAME.value
INPUT_DATA_FOLDER_NAME = FileConfig.INPUT_DATA_FOLDER_NAME.value
CHUNK_SIZE = ModelConfig.CHUNK_SIZE.value
CHUNK_OVERLAP = ModelConfig.CHUNK_OVERLAP.value
MODEL_NAME = ModelConfig.MODEL_NAME.value
TEMPERATURE = ModelConfig.TEMPERATURE.value

# Type definitions
MetadataType = Dict[str, str]
DocumentType = Dict[str, Union[MetadataType, str]]

# --------------------------------------------
# --------- Text Extraction ------------------
# --------------------------------------------

def extract_text_from_pdf() -> str:
    # Extract text from PDF file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, INPUT_DATA_FOLDER_NAME, FILE_NAME)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        pdfreader = PdfReader(file_path)
        text = ''.join(page.extract_text() or '' for page in pdfreader.pages)
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return ""

# --------------------------------------------
# --------- Text Splitting -------------------
# --------------------------------------------

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[DocumentType]:
    # Split the text into smaller chunks
    if not text:
        logger.error("The text to split is empty.")
        raise ValueError("Text cannot be empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

# --------------------------------------------
# --------- Model Initialization -------------
# --------------------------------------------

def create_openai_model(model_name: str = MODEL_NAME, temperature: float = TEMPERATURE) -> ChatOpenAI:
    # Create an instance of the OpenAI model
    return ChatOpenAI(temperature=temperature, model_name=model_name)

# --------------------------------------------
# --------- Text Summarization ---------------
# --------------------------------------------

def summarize_text(chunks: List[DocumentType], model: ChatOpenAI) -> Tuple[Dict[str, str], Dict[str, any]]:
    # Summarize text chunks using the model
    chain = load_summarize_chain(model, chain_type='refine', verbose=False)
    with get_openai_callback() as cb:
        summary = chain.invoke(chunks)
    return summary, cb

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    # Main function to execute the text extraction, splitting, and summarization
    try:
        raw_text = extract_text_from_pdf()
        if not raw_text:
            logger.error("No text extracted from PDF.")
            return
        
        chunks = split_text_into_chunks(raw_text)
        model = create_openai_model()
        summary, callback_info = summarize_text(chunks, model)

        pprint.pprint(summary.get('output_text', 'No summary available.'))
        pprint.pprint(callback_info)
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution flow: {e}")
        raise

if __name__ == "__main__":
    main()
