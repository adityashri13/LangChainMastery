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
    """Returns a list of few-shot examples of meeting notes and their summaries."""
    return [
        {
            "notes": "The team discussed the new project timeline. Decided to extend the deadline by two weeks. Alice will update the project plan. Agreed to have weekly check-ins to monitor progress.",
            "summary": "Project timeline extended by two weeks. Alice will update the project plan. Weekly check-ins agreed upon."
        },
        {
            "notes": "Q2 sales performance was below expectations. The team plans to introduce new incentives and has assigned the sales team to draft a proposal. A follow-up meeting is scheduled for next week to finalize the plan.",
            "summary": "Q2 sales performance was below expectations. New incentives to be introduced. Sales team to draft a proposal. Follow-up meeting next week."
        },
        {
            "notes": "Discussed the Q3 marketing strategy. Agreed to increase the budget by 15%. Decided to focus on social media campaigns. Assigned John to oversee the project.",
            "summary": "Q3 marketing strategy was discussed. Budget increased by 15%. Focus will be on social media campaigns. John assigned to oversee the project."
        },
        {
            "notes": "Reviewed the product roadmap. Decided to prioritize the mobile app launch. Identified a need for additional resources. Sarah will prepare a resource plan.",
            "summary": "Product roadmap review: Mobile app launch prioritized. Additional resources needed. Sarah to prepare a resource plan."
        },
        {
            "notes": "Customer feedback was analyzed, showing a preference for more intuitive UI features. The team agreed to prioritize this in the next sprint. Jane will lead the design overhaul.",
            "summary": "Customer feedback analysis shows preference for intuitive UI. Next sprint will focus on this. Jane to lead design overhaul."
        },
        {
            "notes": "The new hire orientation plan was reviewed. Decided to include more interactive sessions. HR will update the plan accordingly.",
            "summary": "New hire orientation plan updated to include interactive sessions. HR to implement changes."
        }
    ]

# --------------------------------------------
# --------- Create Prompt Template -----------
# --------------------------------------------

def create_example_prompt():
    """Creates a prompt template for few-shot examples using the example data."""
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{notes}"),
        SystemMessagePromptTemplate.from_template("{summary}"),
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
        input_variables=["notes"]
    )

# --------------------------------------------
# --------- Define System Message ------------
# --------------------------------------------

def create_system_message_template():
    """Creates and returns a system message template for the meeting summary assistant."""
    return SystemMessagePromptTemplate(prompt=PromptTemplate(
        template=("You are a meeting assistant that generates concise summaries based on meeting notes. "
                  "Highlight key points, decisions made, and action items. "
                  "Generate a summary from the following notes: {notes}.")
    ))

# --------------------------------------------
# --------- Generate Meeting Summary ----------
# --------------------------------------------

def generate_summary(chat_prompt_template, notes):
    """Generates a meeting summary based on the provided notes."""
    messages = chat_prompt_template.format_messages(notes=notes)

    print('Formatted messages content:')
    for msg in messages:
        print(msg)

    chat_model = ChatOpenAI(model="gpt-4o-mini-turbo", temperature=0.0)
    chat_model_response = chat_model.invoke(messages)

    parser = StrOutputParser()
    formatted_summary = parser.invoke(chat_model_response)
    return formatted_summary

# --------------------------------------------
# ----------------- Main ---------------------
# --------------------------------------------
def main():
    """Main function to drive the execution."""
    examples = get_examples()
    example_prompt = create_example_prompt()
    vectorstore = setup_vector_store(examples)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3,  # Select the top 3 most relevant examples
    )

    few_shot_prompt_template = create_few_shot_prompt_template(example_selector, example_prompt)
    system_message_template = create_system_message_template()

    # Combine the system, few-shot example, and user message templates into a chat prompt template
    chat_prompt_template_with_few_shots = ChatPromptTemplate.from_messages(
        [system_message_template, few_shot_prompt_template]
    )

    # Example meeting notes provided by the user
    notes = "The team discussed the new project timeline. Decided to extend the deadline by two weeks. Alice will update the project plan. Agreed to have weekly check-ins to monitor progress."

    summary = generate_summary(chat_prompt_template_with_few_shots, notes)

    print(f"\nGenerated Summary:\n{summary}")

if __name__ == "__main__":
    main()