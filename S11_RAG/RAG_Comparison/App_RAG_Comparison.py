import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate

# Set Streamlit page layout
st.set_page_config(layout="wide")

# Helper function to load the document
def load_document(file_path):
    try:
        loader = PyPDFLoader(file_path)
        data_docs = loader.load()
        # st.success("Document loaded successfully!")
        return data_docs
    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()

# Helper function to create a vector store and return retrievers
def create_vector_store(data_docs, embedding_model, chunk_sizes, chunk_overlaps):
    retrievers = []
    for chunk_size, chunk_overlap in zip(chunk_sizes, chunk_overlaps):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        # Split the loaded document into chunks
        chunks = text_splitter.split_documents(data_docs)

        # Create embeddings for the chunks
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Create the FAISS vector store
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        retrievers.append(retriever)
    
    return retrievers

# Helper function to run the chain and get responses
def run_chain(retrievers, user_query):
    if user_query:
        prompt_template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        chain = prompt | llm | StrOutputParser()

        # Store each response for the corresponding retriever
        responses = []
        for retriever in retrievers:
            docs = retriever.invoke(user_query)
            response = chain.invoke({"context": docs, "question": user_query})
            responses.append(response)
        return responses

# Helper function to display the responses
def display_responses(responses, chunk_sizes, chunk_overlaps):
    if responses:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"Chunk Size {chunk_sizes[0]} & Overlap {chunk_overlaps[0]}")
            st.write(responses[0])

        with col2:
            st.subheader(f"Chunk Size {chunk_sizes[1]} & Overlap {chunk_overlaps[1]}")
            st.write(responses[1])

        with col3:
            st.subheader(f"Chunk Size {chunk_sizes[2]} & Overlap {chunk_overlaps[2]}")
            st.write(responses[2])

# Main function for the Streamlit app
def main():
    st.title("Ask any question about Sports Personality!")

    # File settings
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = "../data"
    file_name = 'SPORTS_INSPIRATIONAL_STORIES.pdf'
    file_path = os.path.join(script_directory, data_folder, file_name)

    # st.write(f"File Path: {file_path}")

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()

    # Streamlit user inputs
    embedding_model = st.selectbox(
        "Choose an Embedding Model", 
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-3-small"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        chunk_size1 = st.number_input("Chunk Size 1", min_value=50, max_value=10000, value=100)
        chunk_overlap1 = st.number_input("Chunk Overlap 1", min_value=0, max_value=1000, value=20)

    with col2:
        chunk_size2 = st.number_input("Chunk Size 2", min_value=50, max_value=10000, value=100)
        chunk_overlap2 = st.number_input("Chunk Overlap 2", min_value=0, max_value=1000, value=20)

    with col3:
        chunk_size3 = st.number_input("Chunk Size 3", min_value=50, max_value=10000, value=100)
        chunk_overlap3 = st.number_input("Chunk Overlap 3", min_value=0, max_value=1000, value=20)

    chunk_sizes = [chunk_size1, chunk_size2, chunk_size3]
    chunk_overlaps = [chunk_overlap1, chunk_overlap2, chunk_overlap3]

    # Load the document
    data_docs = load_document(file_path)
    
    # Get the user's query
    user_query = st.text_input("Your Question:")

    if st.button("Submit"):
        # Create vector stores and retrievers
        retrievers = create_vector_store(data_docs, embedding_model, chunk_sizes, chunk_overlaps)
        
        # Run the chain and get the answers
        answers = run_chain(retrievers, user_query)

        # Display the responses
        display_responses(answers, chunk_sizes, chunk_overlaps)

# Run the main function
if __name__ == "__main__":
    main()

# who is better cricketer: jadeja or jordan?
# identify cricketers in the context and tell brief about them