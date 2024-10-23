# Import necessary modules and classes from LangChain and OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st  # Import Streamlit
from langchain_core.runnables.passthrough import RunnableAssign

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

# Define the OpenAI model to be used
model = ChatOpenAI(model="gpt-4o-mini")  # Use "gpt-4o-mini" if you have access

# Define a Pydantic model to structure and validate the transcript information
class VideoTranscriptInfo(BaseModel):
    topic: str = Field('general', description="Current conversation topic")
    brand_name: str = Field("", description="Mention of any brand or product in the reel")
    is_offer_pricing_mentioned: bool = Field(False, description="Indicates if any offer or pricing is mentioned")
    review: list = Field([], description="Feedback about the product")

# Function to create a transcript analysis chain
def analyse_transcript(pydantic_class, llm, prompt):
    # Create a Pydantic output parser instance
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    # Define how format instructions are assigned
    format_instructions = RunnableAssign({'format_instructions': lambda x: parser.get_format_instructions()})

    # Return the complete chain for transcript analysis
    return format_instructions | prompt | llm | parser

# Define the prompt template for the transcript analysis
parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "Provide a response based on the given specifications: {format_instructions}. Only use information from the input."),
    ("user", "{input}"),
])

# Create a chain for analyzing video transcripts
video_analysis_chain = analyse_transcript(VideoTranscriptInfo, model, parser_prompt)

# --------------------------------------------
# --------- Streamlit App --------------------
# --------------------------------------------

def main():
    st.title("Video Transcript Analyzer")

    # User input for the transcript
    transcript_input = st.text_area("Enter the video transcript:", height=300)

    if st.button("Analyze Transcript"):
        if not transcript_input.strip():
            st.error("Please enter a video transcript to analyze.")
        else:
            with st.spinner("Analyzing transcript..."):
                # Analyze the video transcript using the defined chain
                video_analysis = video_analysis_chain.invoke({'input': transcript_input})

            # Display the analysis result
            st.subheader("Analysis Results:")
            st.write(video_analysis)

            # Optionally, display individual fields
            st.write(f"**Topic:** {video_analysis.topic}")
            st.write(f"**Brand Name:** {video_analysis.brand_name}")
            st.write(f"**Is Offer/Pricing Mentioned:** {video_analysis.is_offer_pricing_mentioned}")
            st.write(f"**Review:**")
            for item in video_analysis.review:
                st.write(f"- {item}")

if __name__ == "__main__":
    main()
