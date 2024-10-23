# Import necessary modules and classes from LangChain and OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import json
from langchain_core.runnables.passthrough import RunnableAssign

# Define the OpenAI model to be used
model = ChatOpenAI(model="gpt-4o-mini")

# Define a Pydantic model to structure and validate the transcript information
class VideoTranscriptInfo(BaseModel):
    topic: str = Field('general', description="Current conversation topic")
    brand_name: str = Field("", description="Mention of any brand or product in the reel")
    is_offer_pricing_mentioned: bool = Field(False, description="Indicates if any offer or pricing is mentioned")
    review: list = Field([], description="Feedback about the product")

# Print the format instructions for the Pydantic output parser
print(PydanticOutputParser(pydantic_object=VideoTranscriptInfo).get_format_instructions())

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

# Example input transcript for analysis
transcript_input = (
    "Alright guys today, I'm gonna be reviewing prime hydration drink. Before I go any further in this video, "
    "I just want to let you know I'm going to be completely honest. If it's trash, it's trash, especially because "
    "you're not paying me for this. Alright, we're trying the tropical punch one first. Well, I could taste the "
    "coconut water. It's not bad. I like the tropical punch one, pretty good. Next, we got blue raspberry prime. "
    "Now blue raspberry is my favorite flavor, so there's something there that's not supposed to be there. It's a "
    "weird aftertaste. It tastes like medicine. I don't like this one. Oh, last but not least, I got the lemon lime "
    "flavor. It's one of my favorite flavors too, obviously. Oh yeah, my favorite one right here, the lemon lime "
    "flavor, guys, the best one. But yeah guys, that's what I think about prime. I think it's a bit overrated to be "
    "honest. Everybody's hyping it up like it's the best drink in the world. It's definitely nowhere near the best "
    "drink in the world. But if I was working out, I would definitely drink this. Yeah guys, that's my review. Apply "
    "my code DRINK-PRIME and get two bottles for $30.99!!"
)

# Analyze the video transcript using the defined chain
video_analysis = video_analysis_chain.invoke({'input': transcript_input})

# Print the analysis result
print("Analysis:")
print(video_analysis)
print(f"Result Type: {type(video_analysis)}")
