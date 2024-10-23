import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
import json


# Initialize the OpenAI model to be used
model = ChatOpenAI(model="gpt-4o-mini")

# --------------------------------------------
# --------- Content and Hashtag Chain --------
# --------------------------------------------

# Define the prompt template for generating content based on a topic
content_prompt_template = PromptTemplate(
    template="""
    Topic: {topic}
    Content: This is a content for the above topic:
    """,
    input_variables=["topic"]
)

# Define the prompt template for generating hashtags based on content
hashtag_prompt_template = PromptTemplate(
    template="""
    Content:
    {content}
    Generate hashtags for the above content. 
    Please respond strictly in JSON format as:
    {{
        "content": "{content}",
        "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"]
    }}
    """,
    input_variables=["content"]
)

# Wrap the content prompt template in a HumanMessagePromptTemplate
user_message_template_content = HumanMessagePromptTemplate(prompt=content_prompt_template)
user_message_template_hashtags = HumanMessagePromptTemplate(prompt=hashtag_prompt_template)

# Create a ChatPromptTemplate for content
content_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a content creator. Given the topic, it is your job to write short content, which can be spoken in 30 seconds."),
        user_message_template_content
    ]
)

# Define the structure of the output data for content and hashtags
class ContentHashtagOutputStructure(BaseModel):
    """Information about generated content and hashtags."""
    content: str = Field(description="Generated content")
    hashtags: list = Field(description="Generated hashtags")

# Set up a parser for content and hashtags
content_parser = PydanticOutputParser(pydantic_object=ContentHashtagOutputStructure)

# Combine the content and hashtag chains
content_chain = content_chat_prompt_template | model | StrOutputParser()

hashtags_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Generate relevant hashtags for the following content."),
        user_message_template_hashtags
    ]
)

hashtags_chain = hashtags_chat_prompt_template | model | content_parser

# --------------------------------------------
# --------- Platform-Specific Post Chain ----- 
# --------------------------------------------

# Define the prompt template for generating platform-specific posts
platform_post_prompt_template = PromptTemplate(
    template="""
    Platform: {platform}
    Content: {content}
    Hashtags: {hashtags}

    Write a social media post for {platform} using the content and hashtags provided.
    """,
    input_variables=["platform", "content", "hashtags"]
)

# Define the structure of the output data for the final post
class FinalOutputStructure(BaseModel):
    """Information about generated content, hashtags, and platform-specific post."""
    content: str = Field(description="Generated content")
    hashtags: list = Field(description="Generated hashtags")
    post: str = Field(description="Generated platform-specific post")

# Set up a parser for the final output structure
final_output_parser = PydanticOutputParser(pydantic_object=FinalOutputStructure)

# Create a ChatPromptTemplate for platform-specific post
platform_post_chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Create a social media post for a specific platform."),
        HumanMessagePromptTemplate(prompt=platform_post_prompt_template)
    ]
)

# --------------------------------------------
# --------- Overall Chain ------------------- 
# --------------------------------------------

# Define the full chain: generate content and hashtags first, then platform-specific post
def generate_social_media_post(topic, platform):
    # First Chain: Generate content
    content_response = content_chain.invoke({"topic": topic})
    content_output = content_response  # Assume content comes as string directly
    
    # Second: Generate hashtags (ensure content is included in the response)
    hashtags_response = hashtags_chain.invoke({"content": content_output})

    # Access content and hashtags using dot notation
    content_with_hashtags = {
        "content": hashtags_response.content,  # Accessing via dot notation
        "hashtags": hashtags_response.hashtags  # Accessing via dot notation
    }

    # Final Chain: Generate platform-specific post
    final_response = platform_post_chat_prompt_template.invoke(
        {
            "platform": platform,
            "content": content_with_hashtags["content"],
            "hashtags": content_with_hashtags["hashtags"],
        }
    )

    # Extract the generated post content from the final response (access the string output)
    post_text = final_response.messages[-1].content  # Retrieve the last message (model output)
    

    # Prepare final JSON structure manually as the PydanticOutputParser requires string parsing.
    final_output = {
        "content": content_with_hashtags["content"],
        "hashtags": content_with_hashtags["hashtags"],
        "post": post_text
    }
    
    # Convert the final output to JSON for better readability
    json_output = json.dumps(final_output, indent=2)

    # Return the final JSON output
    return final_output


# --------------------------------------------
# --------- Streamlit Integration ------------
# --------------------------------------------

# Set up the Streamlit interface
st.title("Social Media Post Generator")

# Input fields for the user to provide a topic and platform
topic = st.text_input("Enter a topic:")
platform = st.selectbox("Choose a platform:", ["Instagram", "Facebook", "Twitter", "LinkedIn"])

# Button to generate the social media post
if st.button("Generate Post"):
    # Call the function to generate the post
    result = generate_social_media_post(topic, platform)

    # Display the generated content, hashtags, and post in the Streamlit app
    st.subheader("Generated Content")
    st.write(result["content"])
    
    st.subheader("Generated Hashtags")
    st.write(", ".join(result["hashtags"]))
    
    st.subheader(f"Platform-Specific Post for {platform}")
    st.write(result["post"])
