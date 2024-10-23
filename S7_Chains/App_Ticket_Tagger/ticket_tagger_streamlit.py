import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

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
# --------- Streamlit Application ------------ 
# --------------------------------------------

def main():
    st.title("Customer Support Ticket Tagger")

    # Get user input for customer ticket
    ticket_input = st.text_area("Enter Customer Support Ticket:", height=150)

    if st.button("Tag Ticket"):
        if ticket_input:
            try:
                # Initialize the tagging chain
                tagging_chain = get_ticket_tagging_chain()

                # Process the ticket input using the chain
                result = perform_ticket_tagging(tagging_chain, ticket_input)

                # Display the results
                st.subheader("Tagged Ticket Information")
                st.write("**Customer Ticket:**", ticket_input)
                st.write("**Priority:**", result.priority)
                st.write("**Response:**", result.response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a customer ticket before tagging.")

if __name__ == "__main__":
    main()
