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
# --------- Main Function -------------------- 
# --------------------------------------------

def main():
    # Sample hotel reviews
    comments = [
        {"HotelReview": "Horrible experience never go!!!! Employees are rude and can't sleep because the beds are so hard!!!!"},
        {"HotelReview": "My husband and our two small kids decided it would be fun to stay at the Hilton one weekend to swim, eat, and relax. We were quite surprised..."},
        {"HotelReview": "The staff was friendly and polite. The amenities were top-notch."},
        {"HotelReview": "Not bad, but not great either. It was just ok, the location was decent."},
        {"HotelReview": "A very pleasant stay! Highly recommend to anyone visiting the area."},
        {"HotelReview": "El hotel estaba limpio y bien ubicado, pero el servicio fue muy lento y el personal no parecía interesado en ayudar. No creo que vuelva a hospedarme aquí."}
    ]

    try:
        # Initialize the tagging chain
        tagging_chain = get_tagging_chain()

        # Apply the chain to hotel reviews
        for comment in comments:
            result = perform_tagging(tagging_chain, comment["HotelReview"])
            print("-----Quick info of the fields------")
            print("sentiment : The sentiment of the hotel review")
            print("language : The language of the hotel review")
            print("tone : The tone of the hotel review")
            print("response : The response generated based on the tone and sentiment")
            print("------------------------------------------------------")
            pprint(result.dict())  # Convert Pydantic object to dictionary and pretty-print the result
        
    except Exception as e:
        raise RuntimeError(f"An error occurred in the main execution flow: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
