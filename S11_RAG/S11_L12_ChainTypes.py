from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_core.runnables import Runnable

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    # Use LangChain's PyPDFLoader to load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # Load the PDF file as LangChain documents
    # Extract the text from all the loaded documents
    text = ' '.join([doc.page_content for doc in documents])
    return text


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])


def create_vector_store(chunks, embeddings_model):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return Chroma.from_documents(chunks, embeddings)


def generate_response(chain: Runnable, vector_store: Chroma, query: str):
    """Perform question answering based on the provided query and vector store."""

    matching_docs = vector_store.similarity_search(query)

    response = chain.run(input_documents=matching_docs, question=query)

    return response


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = 'input'
    file_name = 'SPORTS_INSPIRATIONAL_STORIES.pdf'
    pdf_path = os.path.join(script_directory, data_folder, file_name)

    raw_text = extract_text_from_pdf(pdf_path)

    # Define model and chain
    model = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

    chain = load_qa_chain(model, chain_type="stuff", verbose=True)

    # Split text into chunks
    chunks = split_text_into_chunks(raw_text)

    # Create vector store
    vector_store = create_vector_store(chunks, "text-embedding-3-small")
    
    # Continuously ask questions until the user decides to exit
    while True:
        query = input("Please enter the query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the question-answering loop. Goodbye!")
            break
        
        answer = generate_response(chain, vector_store, query)
        print(answer)

# Entry point of the script
if __name__ == "__main__":
    main()
