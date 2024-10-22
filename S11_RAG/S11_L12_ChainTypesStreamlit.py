import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# --------- Text Extraction ------------------
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return ' '.join([doc.page_content for doc in documents])

# --------- Text Splitting -------------------
def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

# --------- Vector Store Creation ------------
def create_vector_store(chunks, embeddings_model):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return Chroma.from_documents(chunks, embeddings)

def generate_response(vector_store: Chroma, query: str):
    """Perform question answering based on the provided query and vector store."""
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = vector_store.similarity_search(query)
    response = chain.run(input_documents=matching_docs, question=query)
    return response

# --------- Main Streamlit App ---------------
def main():
    st.title("PDF Question-Answering App")

    # Default file settings
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = 'data'
    default_file_name = 'SPORTS_INSPIRATIONAL_STORIES.pdf'
    default_pdf_path = os.path.join(script_directory, data_folder, default_file_name)

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        uploaded_file_path = os.path.join(script_directory, uploaded_file.name)
        st.write(f"Using uploaded file: {uploaded_file.name}")
    else:
        uploaded_file_path = default_pdf_path
        st.write(f"No file uploaded. Using default file: {os.path.basename(uploaded_file_path)}")

    raw_text = extract_text_from_pdf(uploaded_file_path)

    if raw_text:
        st.success("Text extraction completed!")

        # Split the text into manageable chunks
        st.write("Splitting text into chunks...")
        chunks = split_text_into_chunks(raw_text)
        st.success("Text splitting into chunks completed!")

        # Create vector store
        st.write("Creating vector store...")
        vector_store = create_vector_store(chunks, "text-embedding-3-small")

        # User input for query
        query = st.text_input("Enter your query:")

        if query:
            st.write("Processing your query...")
            answer = generate_response(vector_store, query)
            st.write("Answer:", answer)
    else:
        st.error(f"Text extraction failed. Please try again.")


# Entry point for Streamlit
if __name__ == "__main__":
    main()
