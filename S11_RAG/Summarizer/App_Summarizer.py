import os
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback

# --------- PDF Text Extraction --------------
def extract_text_from_pdf(pdf_file_path) -> list:
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    return [doc.page_content for doc in documents]

# --------- Text Splitting -------------------
def split_text_into_chunks(text: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    chunks = text_splitter.create_documents([text])
    return chunks

# --------- Summarize text -------------------
def summarize_text(chunks: list, model: ChatOpenAI):
    chain = load_summarize_chain(model, chain_type='refine', verbose=False)
    summary = chain.invoke(chunks)
    return summary

# --------- Orchestrate (main function) --------
def main():
    # Specify the PDF file path
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # pdf_file_path = os.path.join(script_directory, 'input', 'Indian_Budget.pdf')
    pdf_file_path = os.path.join(script_directory, '../data/olympics', 'Olympic_History_Part_1.pdf')

    # Check if the file exists
    if not os.path.exists(pdf_file_path):
        print(f"Error: The file {pdf_file_path} does not exist.")
        return

    print("Extracting text from PDF...")

    # Extract text from the PDF
    raw_text = ' '.join(extract_text_from_pdf(pdf_file_path))
    print("Text extraction completed!")

    # Split the text into manageable chunks
    chunks = split_text_into_chunks(raw_text)
    print("Text splitting into chunks completed!")

    # Create the OpenAI model and summarize
    model = ChatOpenAI(temperature=0.5, model_name='gpt-4')
    print("Summarizing the text...")
    summary = summarize_text(chunks, model)

    # Display the summary
    print("\n---------- Summary ----------")
    print(summary.get('output_text', 'No summary available.'))

    
if __name__ == "__main__":
    main()
