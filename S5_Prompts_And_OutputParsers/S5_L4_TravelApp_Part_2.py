from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Define few-shot examples for the prompt template
examples = [
    {
        "question": "How to get from Delhi Airport to India Gate?",
        "answer": """
        option 1: mode=bus, min_time_in_min=60, max_time_in_min=75, description=Take bus number 101 from the airport.
        It will drop you near India Gate. Take a 10-minute walk.
        option 2: mode=metro, min_time_in_min=30, max_time_in_min=45, description=Use the metro rail service from the airport.
        It will bring you to India Gate within 45 minutes.
        option 3: mode=taxi, min_time_in_min=20, max_time_in_min=30, description=Hail a taxi from the airport. It will take
        you directly to India Gate.
        option 4: mode=walk, min_time_in_min=400, max_time_in_min=500, description=Enjoy a leisurely walk from the airport
        to India Gate, which takes around 500 minutes.
        """,
    },
    {
        "question": "How to get from Mumbai Airport to Marine Drive?",
        "answer": """
        option 1: mode=bus, min_time_in_min=45, max_time_in_min=60, description=Take a public bus from the airport to reach
        Marine Drive in 45 to 60 minutes.
        option 2: mode=taxi, min_time_in_min=25, max_time_in_min=35, description=Book a taxi service or use a ride-sharing
        app like Uber or Ola for a comfortable and quick journey to Marine Drive.
        option 3: mode=auto, min_time_in_min=30, max_time_in_min=40, description=Hire an auto-rickshaw from the airport for
        a unique and affordable travel experience to Marine Drive.
        """,
    },
    {
        "question": "How to get from Jaipur Airport to Amber Fort?",
        "answer": """
        option 1: mode=bus, min_time_in_min=40, max_time_in_min=55, description=Take a local bus from the airport to reach
        Amber Fort in 40 to 55 minutes.
        option 2: mode=tuk-tuk, min_time_in_min=25, max_time_in_min=35, description=Hire a tuk-tuk (auto-rickshaw) for a
        fun and adventurous ride from the airport to Amber Fort.
        """,
    },
]

# Create a prompt template for few-shot examples using the example data
example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{question}"),
    SystemMessagePromptTemplate.from_template("{answer}"),
])

# Define a few-shot prompt template with examNow, ples
few_shot_example_template = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["question"]
)

# Create a prompt template for the system message
system_prompt_template = PromptTemplate(
    template="You are a travel expert, helping users to find efficient and comfortable ways of traveling from point A to point B."
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)


# Create a prompt template for the user message
user_prompt_template = PromptTemplate(
    template="How to reach {target} from {source}?",
    input_variables=["target", "source"]
)

# Wrap the user prompt in a HumanMessagePromptTemplate
user_message_template = HumanMessagePromptTemplate(prompt=user_prompt_template)

# Combine the system, few-shot example, and human message templates into a chat prompt template
chat_prompt_template_with_few_shots = ChatPromptTemplate.from_messages(
    [system_message_template, 
     few_shot_example_template,
     user_message_template]
)

# Example user input
user_input_source = "Hawa Mahal"
user_input_target = "Indore Airport"

# Format the messages by replacing the placeholders with actual values
messages = chat_prompt_template_with_few_shots.format_messages(source=user_input_source, target=user_input_target)

# Print the formatted messages' content only
print('Formatted messages content:')
for msg in messages:
    print(msg)

# Create an instance of the ChatOpenAI model
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Generate the response from the chat model using the messages
chat_model_response = chat_model.invoke(messages)

# Initialize a string output parser to process the model's response
parser = StrOutputParser()

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the generated response
print(f"Answer: {formatted_response}")