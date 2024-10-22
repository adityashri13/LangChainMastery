import streamlit as st
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------
# --------- Define Few-Shot Examples ---------
# --------------------------------------------

def get_examples():
    """Returns a list of few-shot examples."""
    return [
        {
            "email": "Dear support, I am facing an issue with my order. The delivery is delayed, and I need it urgently. Can you please help?",
            "response": "Dear Customer, we sincerely apologize for the delay in your order. We are looking into this matter with high urgency and will ensure that it reaches you at the earliest. Thank you for your patience."
        },
        {
            "email": "Hi, I noticed that my bill seems incorrect. Can someone assist me in verifying the charges?",
            "response": "Hello, thank you for reaching out. We apologize for any confusion caused by your bill. Please allow us some time to review the charges, and we will get back to you shortly with a detailed explanation."
        },
        # ... (include the rest of your examples here)
    ]

# --------------------------------------------
# --------- Create Prompt Template -----------
# --------------------------------------------

def create_example_prompt():
    """Creates a prompt template for few-shot examples using the example data."""
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{email}"),
        SystemMessagePromptTemplate.from_template("{response}"),
    ])

# --------------------------------------------
# --------- Setup Vector Store ---------------
# --------------------------------------------

def setup_vector_store(examples):
    """Converts the examples into vectors for semantic similarity search and returns the vectorstore."""
    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()
    return Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# --------------------------------------------
# --------- Setup Few-Shot Prompt ------------
# --------------------------------------------

def create_few_shot_prompt_template(example_selector, example_prompt):
    """Defines and returns a few-shot prompt template with examples."""
    return FewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        input_variables=[
            "email_text", "response_length", "tone", 
            "personalization_level", "language_complexity", 
            "urgency", "additional_instructions"
        ]
    )

# --------------------------------------------
# --------- Define System Message ------------
# --------------------------------------------

def create_system_message_template():
    """Creates and returns a system message template for the email assistant."""
    return SystemMessagePromptTemplate(prompt=PromptTemplate(
        template=("You are an email assistant that generates responses based on user inputs "
                  "regarding response length, tone, personalization, complexity, and urgency. "
                  "Generate a {response_length} response with a {tone} tone, "
                  "personalization level {personalization_level}, "
                  "language complexity {language_complexity}, "
                  "and urgency level {urgency}. "
                  "{additional_instructions}")
    ))

# --------------------------------------------
# --------- Generate Email Response ----------
# --------------------------------------------

def generate_response(chat_prompt_template, email_text, response_length, tone, personalization_level, language_complexity, urgency, additional_instructions):
    """Generates an email response based on the provided parameters."""
    messages = chat_prompt_template.format_messages(
        email_text=email_text,
        response_length=response_length,
        tone=tone,
        personalization_level=personalization_level,
        language_complexity=language_complexity,
        urgency=urgency,
        additional_instructions=additional_instructions
    )

    # Initialize the chat model
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chat_model_response = chat_model.invoke(messages)

    parser = StrOutputParser()
    formatted_response = parser.invoke(chat_model_response)
    return formatted_response

# --------------------------------------------
# ----------------- Main ---------------------
# --------------------------------------------

def main():
    """Main function to drive the execution."""
    st.title("Email Response Generator")

    # Get user inputs
    email_text = st.text_area("Customer Email:")
    response_length = st.selectbox("Response Length:", ["short", "medium", "long"])
    tone = st.selectbox("Tone:", ["Formal", "Semi-Formal", "Informal", "Casual", "Friendly", "Professional"])
    personalization_level = st.selectbox("Personalization Level:", ["High", "Medium", "Low"])
    language_complexity = st.selectbox("Language Complexity:", ["Simple", "Intermediate", "Advanced"])
    urgency = st.selectbox("Urgency Level:", ["High", "Medium", "Low"])
    additional_instructions = st.text_area("Additional Instructions:", value="", height=100)

    if st.button("Generate Response"):
        if not email_text.strip():
            st.error("Please enter the customer email.")
        else:
            examples = get_examples()
            example_prompt = create_example_prompt()
            vectorstore = setup_vector_store(examples)

            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore,
                k=2,  # Select the top 2 most relevant examples
            )

            few_shot_prompt_template = create_few_shot_prompt_template(example_selector, example_prompt)
            system_message_template = create_system_message_template()

            # Combine the system, few-shot example, and user message templates into a chat prompt template
            chat_prompt_template_with_few_shots = ChatPromptTemplate.from_messages(
                [system_message_template, few_shot_prompt_template]
            )

            # Generate response
            with st.spinner('Generating response...'):
                response = generate_response(
                    chat_prompt_template_with_few_shots,
                    email_text,
                    response_length,
                    tone,
                    personalization_level,
                    language_complexity,
                    urgency,
                    additional_instructions
                )

            st.subheader("Generated Response:")
            st.write(response)

if __name__ == "__main__":
    main()
