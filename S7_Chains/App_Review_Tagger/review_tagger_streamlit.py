import os
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st  # Import Streamlit

# --------------------------------------------
# --------- Setup and Configuration ---------- 
# --------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Set model configuration values directly
MODEL_NAME = "gpt-4o-mini"  # Specify the model you want to use
TEMPERATURE = 0.7      # Temperature for response variability

# Initialize OpenAI LLM
def initialize_openai_model() -> ChatOpenAI:
    return ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME)

llm = initialize_openai_model()

# --------------------------------------------
# --------- Pydantic Model for Tagging --------
# --------------------------------------------

class TaggingOutput(BaseModel):
    """Pydantic model to represent the sentiment, language, tone, and response output."""
    sentiment: str = Field(description="The sentiment of the hotel review")
    language: str = Field(description="The language of the hotel review")
    tone: str = Field(description="The tone of the hotel review")
    response: str = Field(description="The response generated based on the tone and sentiment")

# --------------------------------------------
# --------- Tagging and Response Chain Creation ------------ 
# --------------------------------------------

def get_tagging_chain() -> RunnableParallel:
    # Define the schema to ask for tone, sentiment, language, and generate a response
    schema_template = """Tag the sentiment, language, and tone for a hotel review based on the following:
    - Sentiment: Positive, Negative, Neutral, Abusive, etc.
    - Language: Chinese, Spanish, English, Hindi, Arabic, etc.
    - Tone: Formal, Informal, Sarcastic, Angry, Friendly, etc.
    
    Review: {HotelReview}
    
    Please respond in JSON format:
    {{
      "sentiment": "sentiment_value",
      "language": "language_value",
      "tone": "tone_value",
      "response": "response_value"  # Provide a response appropriate to the detected tone and sentiment
    }}
    """

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(schema_template)

    # Create a runnable chain using RunnableParallel
    chain = (
        RunnableParallel({"HotelReview": RunnablePassthrough()})
        | prompt
        | llm
    )
    
    return chain

# --------------------------------------------
# --------- Question Answering --------------- 
# --------------------------------------------

class HotelReview(BaseModel):
    __root__: str

def perform_tagging(tagging_chain: RunnableParallel, review: str) -> TaggingOutput:
    # Run the tagging chain to perform sentiment, language, tone tagging, and response generation
    if not review:
        raise ValueError("Hotel review cannot be empty.")
    
    chain = tagging_chain.with_types(input_type=HotelReview)
    review_input = HotelReview(__root__=review)
    
    # Execute the chain
    response = chain.invoke(review_input.__root__)
    
    # Define the output parser for TaggingOutput model
    parser = PydanticOutputParser(pydantic_object=TaggingOutput)
    
    # Parse the response content into the Pydantic model
    return parser.parse(response.content)

# --------------------------------------------
# --------- Streamlit Integration ------------- 
# --------------------------------------------

def main():
    # Initialize the tagging chain
    tagging_chain = get_tagging_chain()

    # Streamlit UI
    st.title("Hotel Review Sentiment, Language, and Tone Analysis")

    st.write("Enter a hotel review below to analyze its sentiment, language, tone, and to receive an appropriate response.")

    # Text input for hotel review
    review_input = st.text_area("Hotel Review Input", height=150)

    if st.button("Analyze Review"):
        if review_input:
            try:
                # Perform tagging and analysis
                result = perform_tagging(tagging_chain, review_input)

                # Display results in Streamlit
                st.subheader("Analysis Results")
                st.write(f"**Language:** {result.language}")
                st.write(f"**Sentiment:** {result.sentiment}")
                st.write(f"**Tone:** {result.tone}")
                st.write(f"**Response:** {result.response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please input a hotel review before analyzing.")

if __name__ == "__main__":
    main()
