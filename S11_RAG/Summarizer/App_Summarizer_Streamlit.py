import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --------- Unified Text Extraction for PDF and TXT --------------
def extract_text(file_path: str, file_extension: str) -> str:
    """Extract text from a file, whether it's PDF or TXT."""
    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == "txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    documents = loader.load()
    return ' '.join([doc.page_content for doc in documents])

# --------- Text Splitting -------------------
def split_text_into_chunks(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([text])
    return chunks

# --------- Summarize text -------------------
def summarize_text(chunks: list, model: ChatOpenAI):
    chain = load_summarize_chain(model, chain_type='refine', verbose=False)
    summary = chain.invoke(chunks)
    return summary

# --------- Orchestrate (main function) --------
def main():
    st.title("Summarization App")

    # Allow both PDF and TXT file uploads
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    # Default file paths
    script_directory = os.path.dirname(os.path.abspath(__file__))
    default_txt_path = os.path.join(script_directory, '../data/olympics', 'Olympics_2024.txt')
    default_pdf_path = os.path.join(script_directory, '../data/olympics', 'Olympic_History_Part_1.pdf')

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        file_extension = uploaded_file.name.split(".")[-1].lower()
        uploaded_file_path = os.path.join(script_directory, uploaded_file.name)
        
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.write(f"Using uploaded file: {uploaded_file.name}")
    else:
        # If no file is uploaded, use a default file (PDF or TXT)
        file_extension = "txt"  # Change here to switch between 'pdf' or 'txt' based on your use case
        uploaded_file_path = default_txt_path if file_extension == "txt" else default_pdf_path
        st.write(f"No file uploaded. Using default file: {os.path.basename(uploaded_file_path)}")

    # Inputs for chunk size and chunk overlap
    chunk_size = st.number_input("Chunk Size", min_value=50, max_value=10000, value=250)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=50)

    # Run on Submit
    if st.button("Submit"):
        try:
            st.write(f"Extracting text from {file_extension.upper()} file...")
            raw_text = extract_text(uploaded_file_path, file_extension)

            if raw_text:
                st.success("Text extraction completed!")

                # Split the text into manageable chunks based on user input
                st.write("Splitting text into chunks...")
                chunks = split_text_into_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success("Text splitting into chunks completed!")

                # Create the OpenAI model
                model = ChatOpenAI(temperature=0.5, model_name="gpt-4")

                # Summarize the text
                st.write("Summarizing the text...")
                summary = summarize_text(chunks, model)

                # Display the summary
                st.subheader("Summary")
                st.write(summary.get('output_text', 'No summary available.'))
            else:
                st.error("Text extraction failed. Please try again with a different file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
