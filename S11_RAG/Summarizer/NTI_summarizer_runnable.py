import logging
import os
from typing import List, Tuple, Dict, Union
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
import pprint
from config import FileConfig, ModelConfig
from S11_RAG.Budget_Summarizer.NTI_logging_config import setup_logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables
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

class TextInput(BaseModel):
    file_path: str

# --------------------------------------------
# --------- Text Extraction ------------------
# --------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    # Extract text from PDF file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, INPUT_DATA_FOLDER_NAME, file_path)
    
    if not os.path.exists(full_path):
        logger.error(f"File not found: {full_path}")
        raise FileNotFoundError(f"File not found: {full_path}")

    try:
        pdfreader = PdfReader(full_path)
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
    if chunk_size <= 0:
        logger.error("Chunk size must be positive.")
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        logger.error("Chunk overlap must be non-negative.")
        raise ValueError("chunk_overlap must be non-negative")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([text])
    return chunks

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
    if not chunks or not isinstance(chunks, list):
        logger.error("Invalid chunks provided for summarization.")
        raise ValueError("Invalid chunks provided.")
    if not isinstance(model, ChatOpenAI):
        logger.error("Invalid model instance provided.")
        raise ValueError("Invalid model instance provided.")
    
    chain = load_summarize_chain(model, chain_type='refine', verbose=False)
    with get_openai_callback() as cb:
        summary = chain.invoke(chunks)
    return summary, cb

class Summarizer(BaseModel):
    file_name: str
    chunk_size: int
    chunk_overlap: int
    model_name: str
    temperature: float

def create_rag_chain(model: ChatOpenAI, text: str) -> RunnableParallel:
    # Create a Retrieval-Augmented Generation (RAG) chain
    chunks = split_text_into_chunks(text)
    
    chain = (
        RunnableParallel({
            "content_type": RunnablePassthrough(),
            "topic": RunnablePassthrough(),
            "title": RunnablePassthrough(),
            "category": RunnablePassthrough(),
            "audience": RunnablePassthrough(),
            "format": RunnablePassthrough(),
            "duration": RunnablePassthrough()
        })
        | model
        | StrOutputParser()
    )
    
    return chain

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    # Main function to execute the text extraction, splitting, embedding creation, and question answering
    try:
        # Load configurations
        summarizer = Summarizer(
            file_name=FILE_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE
        )

        raw_text = extract_text_from_pdf(summarizer.file_name)
        if not raw_text:
            logger.error("No text extracted from PDF.")
            return

        model = create_openai_model(summarizer.model_name, summarizer.temperature)
        chunks = split_text_into_chunks(raw_text, summarizer.chunk_size, summarizer.chunk_overlap)
        summary, callback_info = summarize_text(chunks, model)

        pprint.pprint(summary.get('output_text', 'No summary available.'))
        pprint.pprint(callback_info)
        
    except Exception as e:
        logger.error(f"An error occurred in the main execution flow: {e}")
        raise

if __name__ == "__main__":
    main()
