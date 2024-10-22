import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------
# --------- Text Extraction ------------------
# --------------------------------------------

def extract_text_from_pdf(file_name, input_data_folder):
    # Extract text from PDF file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_directory, input_data_folder, file_name)
    
    loader = PyPDFLoader(full_path)
    data_docs = loader.load()
    text = ''.join([doc.page_content for doc in data_docs])
    return text

# --------------------------------------------
# --------- Text Splitting -------------------
# --------------------------------------------

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# --------------------------------------------
# --------- Text Summarization --------------- 
# --------------------------------------------

def summarize_chunks_in_parts(chunks, model, max_tokens=4000):
    # Summarize text chunks in parts to avoid hitting the token limit
    template = """Summarize the following text:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Summarize each chunk separately and combine the summaries
    summaries = []
    current_tokens = 0
    chunk_part = []
    
    for chunk in chunks:
        # Estimate token count of the current chunk
        token_count = len(chunk.split())
        
        # If adding this chunk exceeds the max tokens limit, process the accumulated chunks
        if current_tokens + token_count > max_tokens:
            context = "\n".join(chunk_part)
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )
            summary = rag_chain.invoke({"context": context, "question": "Summarize the content."})
            summaries.append(summary)
            
            # Reset for the next batch of chunks
            chunk_part = []
            current_tokens = 0
        
        # Add the current chunk to the batch
        chunk_part.append(chunk)
        current_tokens += token_count
    
    # Process any remaining chunks
    if chunk_part:
        context = "\n".join(chunk_part)
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        summary = rag_chain.invoke({"context": context, "question": "Summarize the content."})
        summaries.append(summary)
    
    # Combine all summaries into one final summary
    final_summary = "\n".join(summaries)
    return final_summary

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main():
    # Extract text from the PDF directly with inline arguments
    raw_text = extract_text_from_pdf("Indian_Budget.pdf", "input")

    # Split the text into chunks with inline chunk size and overlap
    chunks = split_text_into_chunks(raw_text, chunk_size=3000, chunk_overlap=500)

    # Initialize the model
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

    # Summarize the text chunks in parts to avoid hitting token limits
    summary = summarize_chunks_in_parts(chunks, model, max_tokens=1000)

    # Print the summary
    print("Summary of the document:")
    print(summary)

if __name__ == "__main__":
    main()
