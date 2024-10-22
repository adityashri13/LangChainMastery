import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set model configuration values
MODEL_NAME = "gpt-4o-mini"  # Model name for OpenAI LLM
TEMPERATURE = 0.2      # Lower temperature for more deterministic outputs (useful for SQL generation)

# Initialize OpenAI LLM
def initialize_openai_model() -> ChatOpenAI:
    """Initialize and return the OpenAI language model."""
    return ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)

llm = initialize_openai_model()

# --------------------------------------------
# --------- SQL Generation Chain -------------
# --------------------------------------------

def get_sql_generation_chain() -> RunnableParallel:
    """
    Create a chain that converts a natural language question to SQL.
    The user inputs the schema, table, and natural language question.
    """
    # Define the template for generating SQL code with user input for schema and table
    template = """
    Convert the following natural language question into an SQL query.
    Use the table {schema_name}.{table_name}:
    
    Question: {english_question}
    
    SQL Query:
    """

    # Define the prompt template using v2's ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)

    # Create a runnable chain using RunnableParallel
    chain = (
        RunnableParallel({
            "schema_name": RunnablePassthrough(),
            "table_name": RunnablePassthrough(),
            "english_question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# --------------------------------------------
# --------- User Input and SQL Generation ----
# --------------------------------------------

def get_user_inputs() -> Dict[str, str]:
    """
    Get user inputs for the schema name, table name, and natural language question.
    Returns a dictionary containing these inputs.
    """
    schema_name = input("Enter the schema name: ")
    table_name = input("Enter the table name: ")
    english_question = input("Enter your English question: ")

    return {
        "schema_name": schema_name,
        "table_name": table_name,
        "english_question": english_question,
    }

def perform_sql_generation(sql_chain: RunnableParallel, user_inputs: Dict[str, str]) -> Any:
    """
    Generate SQL query using the provided chain and user inputs.
    """
    return sql_chain.invoke(user_inputs)

# --------------------------------------------
# --------- Main Function --------------------
# --------------------------------------------

def main() -> None:
    """
    Main function to execute the SQL generation process.
    It gets user inputs, runs the SQL generation chain, and prints the output.
    """
    try:
        # Initialize the SQL generation chain
        sql_generation_chain = get_sql_generation_chain()

        # Get user inputs
        user_inputs = get_user_inputs()

        # Generate SQL query
        sql_query = perform_sql_generation(sql_generation_chain, user_inputs)

        print(f"\nGenerated SQL Query: \n{sql_query}")

    except Exception as e:
        raise RuntimeError(f"An error occurred in the main execution flow: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
