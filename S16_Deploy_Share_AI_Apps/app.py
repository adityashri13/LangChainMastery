import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------
# --------- Setup and Configuration ----------
# --------------------------------------------

def initialize_openai_model():
    """Initialize the OpenAI model with the API key from the environment."""
    return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini-0125")

llm = initialize_openai_model()

# --------------------------------------------
# --------- Define Prompt Template -----------
# --------------------------------------------

def define_prompt_template():
    """Define the prompt template for generating social media scripts."""
    return ChatPromptTemplate.from_messages([
        ("user", """You are a social media script writer. 
                    Write a {format} script for {content_type} about {topic}
                    the category of the topic is {category}, the target audience includes {audience}, and
                    the duration is {duration} seconds.
                    Generate appropiate title as well for the generated content.

                    Output format:
                    Title:
                    Script:
         """
                 ),
    ])

prompt = define_prompt_template()

# Create the chain
chain = prompt | llm | StrOutputParser()

# --------------------------------------------
# --------- Script Generation Function -------
# --------------------------------------------

def generate_script(content_type, topic, category, audience, format, duration):
    """Generate a script for social media using LangChain."""
    response = chain.invoke({
        "content_type": content_type,
        "topic": topic,
        "category": category,
        "audience": ', '.join(audience),
        "format": format,
        "duration": duration
    })
    return response

# --------------------------------------------
# --------- Streamlit App --------------------
# --------------------------------------------

def main():
    """Main function to interact with the user and generate a social media script using Streamlit."""
    st.title("Social Media Script Generator")
    st.write("This app generates scripts for social media content using OpenAI's gpt-4o-mini-0125 model.")

    # UI for content type selection
    content_type = st.selectbox("Select Content Type", ["YouTube", "Instagram"]).lower()

    # UI for topic input
    topic = st.text_input("Enter the Topic for the Script")

    # UI for category input with predefined options
    category = st.selectbox("Select the Category of the Script", ["Fashion", "Education", "Technology", "Health", "Travel"])

    # UI for target audience selection with multiple select checkboxes
    audience = st.multiselect(
        "Select the Target Audience",
        ["Students", "Housewives", "Professionals", "Teenagers", "Seniors"]
    )

    # UI for format selection
    format = st.selectbox("Select Format", ["Reel", "Video"]).lower()

    # UI for duration input using a slider
    duration = st.slider("Select the Duration of the Script in Seconds", min_value=15, max_value=300, step=5)

    if st.button("Generate Script"):
        if content_type and category and audience and format and duration and topic:
            try:
                with st.spinner("Generating script..."):
                    script = generate_script(content_type, topic, category, audience, format, duration)
                st.success("Script generated successfully!")
                st.subheader("Generated Script")
                st.write(script)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill in all the fields.")

# Entry point of the Streamlit app
if __name__ == "__main__":
    main()


# ##### app.py #####


# import streamlit as st
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# import os

# # --------------------------------------------
# # --------- Setup and Configuration ----------
# # --------------------------------------------

# def initialize_openai_model(api_key):
#     """Initialize the OpenAI model with the API key."""
#     os.environ["OPENAI_API_KEY"] = api_key  # Set the API key dynamically
#     return ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini-0125")

# # --------------------------------------------
# # --------- Define Prompt Template -----------
# # --------------------------------------------

# def define_prompt_template():
#     """Define the prompt template for generating social media scripts."""
#     return ChatPromptTemplate.from_messages([
#         ("user", """You are a social media script writer. 
#                     Write a {format} script for {content_type} about {topic}
#                     the category of the topic is {category}, the target audience includes {audience}, and
#                     the duration is {duration} seconds.
#                     Generate appropriate title as well for the generated content.

#                     Output format:
#                     Title:
#                     Script:
#          """
#         ),
#     ])

# prompt = define_prompt_template()

# # Create the chain
# def create_chain(llm):
#     """Create the chain with the LLM model and the output parser."""
#     return prompt | llm | StrOutputParser()

# # --------------------------------------------
# # --------- Script Generation Function -------
# # --------------------------------------------

# def generate_script(content_type, topic, category, audience, format, duration, chain):
#     """Generate a script for social media using LangChain."""
#     response = chain.invoke({
#         "content_type": content_type,
#         "topic": topic,
#         "category": category,
#         "audience": ', '.join(audience),
#         "format": format,
#         "duration": duration
#     })
#     return response

# # --------------------------------------------
# # --------- Streamlit App --------------------
# # --------------------------------------------

# def main():
#     """Main function to interact with the user and generate a social media script using Streamlit."""
#     st.title("Social Media Script Generator")
#     st.write("This app generates scripts for social media content using OpenAI's gpt-4o-mini-0125 model.")

#     # Ask the user for the OpenAI API key
#     api_key = st.text_input("Enter your OpenAI API key", type="password")

#     if api_key:
#         # Initialize the OpenAI model with the provided API key
#         llm = initialize_openai_model(api_key)
#         chain = create_chain(llm)

#         # UI for content type selection
#         content_type = st.selectbox("Select Content Type", ["YouTube", "Instagram"]).lower()

#         # UI for topic input
#         topic = st.text_input("Enter the Topic for the Script")

#         # UI for category input with predefined options
#         category = st.selectbox("Select the Category of the Script", ["Fashion", "Education", "Technology", "Health", "Travel"])

#         # UI for target audience selection with multiple select checkboxes
#         audience = st.multiselect(
#             "Select the Target Audience",
#             ["Students", "Housewives", "Professionals", "Teenagers", "Seniors"]
#         )

#         # UI for format selection
#         format = st.selectbox("Select Format", ["Reel", "Video"]).lower()

#         # UI for duration input using a slider
#         duration = st.slider("Select the Duration of the Script in Seconds", min_value=15, max_value=300, step=5)

#         if st.button("Generate Script"):
#             if content_type and category and audience and format and duration and topic:
#                 try:
#                     with st.spinner("Generating script..."):
#                         script = generate_script(content_type, topic, category, audience, format, duration, chain)
#                     st.success("Script generated successfully!")
#                     st.subheader("Generated Script")
#                     st.write(script)
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#             else:
#                 st.warning("Please fill in all the fields.")
#     else:
#         st.warning("Please enter your OpenAI API key.")

# # Entry point of the Streamlit app
# if __name__ == "__main__":
#     main()
