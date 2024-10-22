# Install the necessary packages using the following commands:
# pip install langchain-openai PyPDF2 streamlit

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.callbacks import get_openai_callback

# Set your OpenAI API key
OPENAI_API_KEY = "your_openai_api_key_here"

# --------------------------------------------
# --------- Initialize OpenAI Model ----------
# --------------------------------------------

# Initialize the OpenAI Model
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5, model_name="gpt-4o-mini")

# --------------------------------------------
# --------- Streamlit UI Setup ---------------
# --------------------------------------------

st.title("PDF Summarization Tool")
uploaded_file = st.file_uploader("Upload your PDF document:", type="pdf")

# --------------------------------------------
# --------- PDF Text Extraction Function ------
# --------------------------------------------

def extract_text_from_pdf(pdf_file):
    """Extract text from the uploaded PDF file."""
    try:
        pdfreader = PdfReader(pdf_file)
        text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --------------------------------------------
# --------- Summarization Function -----------
# --------------------------------------------

def summarize_text(chunks):
    """Summarize the extracted text from the PDF."""
    with get_openai_callback() as cb:
        summary = chain.invoke(chunks)
    return summary['output_text'], cb

# --------------------------------------------
# --------- Main Application Logic -----------
# --------------------------------------------
# TODO: create main function
# TODO: put proper comments in code everywhere
if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    if raw_text:
        # Split text into manageable chunks
        # TODO: create numeric inputs for chunk_size, chunk_overlap for users to play around with them
        # TODO: create slider for temperature for users to play around with it
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        chunks = text_splitter.create_documents([raw_text])

        # Load the summarization chain
        chain = load_summarize_chain(llm, chain_type='refine', verbose=False)

        # Button to trigger summarization
        if st.button('Summarize'):
            summary, callback_info = summarize_text(chunks)
            st.write("### Summary:")
            st.write(summary)
            st.write("### Callback Information:")
            st.json(callback_info)


# TODO: create another area where we can see two columns i.e. user can input chunk_size & chunk_overlap for each column
# and can see how output changes with chunk_size & chunk_overlap values. Create a menu for this named 'Compare outputs'
