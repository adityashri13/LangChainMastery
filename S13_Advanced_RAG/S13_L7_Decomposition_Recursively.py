import os
import pprint
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.load import dumps, loads
from langchain_core.runnables import RunnableMap, RunnablePassthrough
# from dotenv import load_dotenv

# load_dotenv()
# =====================
# 2. File Setup
# =====================
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the name of the data folder and the file
data_folder = 'data/olympics'
file_name = 'Olympic_History_Part_1.txt'

# Construct the file path
file_path = os.path.join(script_directory, data_folder, file_name)

# =====================
# 3. Load the Data
# =====================
loader = TextLoader(file_path, encoding='UTF-8')
data_docs = loader.load()

# =====================
# 4. Split the Data into Chunks
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Set a small chunk size for demonstration
    chunk_overlap=20,     # Overlap between chunks to maintain context
    length_function=len,  # Function to measure the length of the chunks
    is_separator_regex=False  # Treat the separator as a fixed string
)

# Make splits
splits = text_splitter.split_documents(data_docs)

# =====================
# 5. Create and Index Embeddings
# =====================
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# =====================
# 6. Query Decomposition and Generation
# =====================
# Define decomposition prompt
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = (
    prompt_decomposition 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Run query decomposition
user_query = "what do you know of Panagiotis Soutsos"
questions = generate_queries_decomposition.invoke({"question": user_query})
pprint.pprint(questions)

# =====================
# 7. Answer Generation for Sub-Questions
# =====================
# Define the prompt for generating answers
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
decomposition_prompt = ChatPromptTemplate.from_template(template)

# Function to format Q and A pairs
def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


# Generate answers for each sub-question
q_a_pairs = ""
for q in questions:
    print('---q')
    print(q)
    rag_chain = (
        {"context": itemgetter("question") | retriever, 
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    print('---answer')
    print(answer)
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs += "\n---\n" + q_a_pair

print('-----final answer')
print(answer)


# # Generate answers for each sub-question
# q_a_pairs = ""
# for q in questions:
#     print('---q')
#     print(q)
    
#     # Retrieve relevant documents for the sub-question
#     retrieved_docs = retriever.invoke(q)
    
#     # Convert the retrieved_docs list into a single string of concatenated documents for context
#     context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
#     # Prepare the inputs with context, question, and q_a_pairs
#     inputs = {
#         "context": context,
#         "question": q,
#         "q_a_pairs": q_a_pairs
#     }
    
#     # Create the RAG chain using inputs
#     rag_chain = (
#         decomposition_prompt
#         | ChatOpenAI(temperature=0)
#         | StrOutputParser()
#     )

#     # Generate the answer using the context and the question
#     answer = rag_chain.invoke(inputs)
#     print('---answer')
#     print(answer)
#     q_a_pair = format_qa_pair(q, answer)
#     q_a_pairs += "\n---\n" + q_a_pair

# print('-----final answer')
# print(answer)
