import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

# --------------------------------------------
# --------- Setup and Configuration ---------- 
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set model configuration values
MODEL_NAME = "gpt-4o-mini"  # Specify the model you want to use
TEMPERATURE = 0.7      # Temperature for response variability

# Initialize OpenAI LLM
def initialize_openai_model() -> ChatOpenAI:
    return ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)

llm = initialize_openai_model()

# --------------------------------------------
# --------- Pydantic Model for Tagging --------
# --------------------------------------------

class TicketTaggingOutput(BaseModel):
    """Pydantic model to represent the ticket priority and response output."""
    priority: str = Field(description="The priority of the customer ticket")
    response: str = Field(description="The response generated based on the priority")

# --------------------------------------------
# --------- Tagging and Response Chain Creation ------------ 
# --------------------------------------------

def get_ticket_tagging_chain() -> RunnableParallel:
    # Define the schema to ask for priority tagging and generate a response
    schema_template = """Analyze the content of a customer support ticket and assign a priority:
    - Priority: High, Medium, Low
    Then generate a response based on the priority.

    Ticket: {CustomerTicket}
    
    Please respond in JSON format:
    {{
      "priority": "priority_value",
      "response": "response_value"  # Provide a response appropriate to the priority level
    }}
    """

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(schema_template)

    # Create a runnable chain using RunnableParallel
    chain = (
        RunnableParallel({"CustomerTicket": RunnablePassthrough()})
        | prompt
        | llm
    )
    
    return chain

# --------------------------------------------
# --------- Ticket Tagging Execution ---------- 
# --------------------------------------------

class CustomerTicket(BaseModel):
    __root__: str

def perform_ticket_tagging(tagging_chain: RunnableParallel, ticket: str) -> TicketTaggingOutput:
    # Run the tagging chain to perform priority tagging and response generation
    if not ticket:
        raise ValueError("Customer ticket cannot be empty.")
    
    chain = tagging_chain.with_types(input_type=CustomerTicket)
    ticket_input = CustomerTicket(__root__=ticket)
    
    # Execute the chain
    response = chain.invoke(ticket_input.__root__)
    
    # Define the output parser for TicketTaggingOutput model
    parser = PydanticOutputParser(pydantic_object=TicketTaggingOutput)
    
    # Parse the response content into the Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Main Function -------------------- 
# --------------------------------------------

def main():
    # Sample customer tickets
    tickets = [
        {"CustomerTicket": "I have an urgent issue with my account being locked and need immediate access."},
        {"CustomerTicket": "I would like to change my email address associated with the account."},
        {"CustomerTicket": "I noticed a minor error in my invoice, please correct it when you can."},
        {"CustomerTicket": "The website is down, and I can't access my services! This is very frustrating."},
        {"CustomerTicket": "Please provide me with a copy of my recent transactions for the last month."}
    ]

    try:
        # Initialize the tagging chain
        tagging_chain = get_ticket_tagging_chain()

        # Apply the chain to customer tickets
        for ticket in tickets:
            result = perform_ticket_tagging(tagging_chain, ticket["CustomerTicket"])
            print("-----Quick info of the fields------")
            print("priority : The priority of the customer ticket")
            print("response : The response generated based on the priority")
            print("------------------------------------------------------")
            print(ticket)
            pprint(result.dict())  # Convert Pydantic object to dictionary and pretty-print the result
        
    except Exception as e:
        raise RuntimeError(f"An error occurred in the main execution flow: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
