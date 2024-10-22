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

# Define few-shot examples for generating email responses
examples = [
    {
        "email": "Dear support, I am facing an issue with my order. The delivery is delayed, and I need it urgently. Can you please help?",
        "response": "Dear Customer, we sincerely apologize for the delay in your order. We are looking into this matter with high urgency and will ensure that it reaches you at the earliest. Thank you for your patience."
    },
    {
        "email": "Hi, I noticed that my bill seems incorrect. Can someone assist me in verifying the charges?",
        "response": "Hello, thank you for reaching out. We apologize for any confusion caused by your bill. Please allow us some time to review the charges, and we will get back to you shortly with a detailed explanation."
    },
    {
        "email": "I need to reschedule my appointment. Can you help me with that?",
        "response": "Dear Customer, we have successfully rescheduled your appointment to the requested date. If there are any other changes needed, please let us know. We are here to assist you."
    },
    {
        "email": "Hello, I would like to know if there are any discounts available for bulk purchases.",
        "response": "Dear valued customer, thank you for your inquiry. We are pleased to inform you that we do offer discounts on bulk purchases. Please provide us with the details of your order, and we will be happy to assist you further."
    },
    {
        "email": "Hi, I want to cancel my subscription. How do I proceed?",
        "response": "Dear User, we are sorry to hear that you wish to cancel your subscription. To proceed with the cancellation, please follow the instructions provided in the attached document. If you have any further questions, feel free to contact us."
    },
    {
        "email": "My recent purchase is not working as expected. Could you provide a replacement?",
        "response": "Dear Customer, we regret the inconvenience caused by the faulty product. We will expedite a replacement and ensure it reaches you promptly. Thank you for bringing this to our attention."
    },
    {
        "email": "I’m interested in bulk ordering your products. Could you share any discounts available?",
        "response": "Dear Customer, thank you for your interest in our products. We are pleased to offer a discount on bulk orders. Please let us know your requirements, and we will provide you with the best possible deal."
    },
    {
        "email": "Could you extend the deadline for my project submission? I need more time due to unforeseen circumstances.",
        "response": "Dear User, we understand your situation and have extended the deadline by an additional week. Please make sure to submit your project within this new timeframe. We hope this adjustment helps."
    },
    {
        "email": "My account was overcharged this month. Could you help correct this?",
        "response": "Dear Customer, we sincerely apologize for the overcharge on your account. Our team is investigating the issue, and we will correct the charges as soon as possible. Thank you for your understanding."
    },
    {
        "email": "I need urgent support for the issue I'm facing with your service. It’s affecting my business.",
        "response": "Dear Valued Customer, we understand the urgency of your situation and have escalated your case to our highest priority. Our team is working on a solution, and we will provide you with an update shortly."
    },
    {
        "email": "Can you please provide a simple guide for setting up your software?",
        "response": "Dear User, we have attached a straightforward guide that will help you set up the software with ease. Should you need further assistance, feel free to reach out to our support team."
    },
    {
        "email": "I noticed some discrepancies in the invoice. Could you clarify these charges?",
        "response": "Dear Customer, thank you for bringing this to our attention. We have reviewed your invoice and identified the discrepancies. Please find the corrected invoice attached."
    },
    {
        "email": "I’m having trouble logging into my account. Could you help me reset my password?",
        "response": "Dear User, we have initiated the password reset process for your account. Please check your email for further instructions. If you continue to experience issues, contact our support team."
    },
    {
        "email": "I received a damaged item in my order. What should I do next?",
        "response": "Dear Customer, we apologize for the damaged item you received. Please send us a photo of the item, and we will arrange for a replacement or refund as per your preference."
    },
    {
        "email": "Can I upgrade my current plan? I need more features for my business.",
        "response": "Dear Valued Customer, we are happy to assist you with upgrading your plan. Please review the available options and let us know which one suits your needs best. We’ll process the upgrade promptly."
    },
    {
        "email": "I’m not satisfied with the service I received. Could you address my concerns?",
        "response": "Dear Customer, we are sorry to hear that our service did not meet your expectations. We take your feedback seriously and will work on addressing your concerns immediately. Thank you for letting us know."
    },
    {
        "email": "I have a technical issue with the software. Can someone assist me?",
        "response": "Dear User, our technical team is ready to assist you with your issue. Please provide us with more details, and we will guide you through the necessary steps to resolve the problem."
    },
    {
        "email": "Could you provide me with a detailed report on my account activities for the last month?",
        "response": "Dear Customer, we have prepared a detailed report of your account activities for the last month. Please find the report attached. Let us know if you have any further questions."
    },
    {
        "email": "I would like to cancel my service. Could you confirm this for me?",
        "response": "Dear Customer, we are sorry to see you go. Your service has been canceled as per your request. If you change your mind, we are always here to welcome you back."
    },
    {
        "email": "Could you provide me with an update on the status of my order?",
        "response": "Dear Customer, your order is currently being processed and will be shipped out shortly. We appreciate your patience and will send you a notification once it’s on its way."
    }
]


# Create a prompt template for few-shot examples using the example data
example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{email}"),
    SystemMessagePromptTemplate.from_template("{response}"),
])

# Convert the examples into vectors for semantic similarity search
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# Use SemanticSimilarityExampleSelector to select the most relevant examples
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Select the top 2 most relevant examples
)

# Define a few-shot prompt template with examples
few_shot_example_template = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["email_text", "response_length", "tone", "personalization_level", "language_complexity", "urgency", "additional_instructions"]
)

# Create a prompt template for the system message
system_prompt_template: PromptTemplate = PromptTemplate(
    template=("You are an email assistant that generates responses based on user inputs "
              "regarding response length, tone, personalization, complexity, and urgency. "
              "Generate a {response_length} response with a {tone} tone, "
              "personalization level {personalization_level}, "
              "language complexity {language_complexity}, "
              "and urgency level {urgency}. "
              "{additional_instructions}")
)

# Wrap the system prompt in a SystemMessagePromptTemplate
system_message_template = SystemMessagePromptTemplate(prompt=system_prompt_template)

# Combine the system, few-shot example, and user message templates into a chat prompt template
chat_prompt_template_with_few_shots = ChatPromptTemplate.from_messages(
    [system_message_template, 
     few_shot_example_template]
)

# Example user inputs
email_text = "Dear support, I am facing an issue with my order. The delivery is delayed, and I need it urgently. Can you please help?"
response_length = "medium"  # Options: 'short', 'medium', 'long'
tone = "Professional"  # Options: 'Formal', 'Semi-Formal', 'Informal', 'Casual', 'Friendly', 'Professional'
personalization_level = "High"  # Options: 'High', 'Medium', 'Low'
language_complexity = "Intermediate"  # Options: 'Simple', 'Intermediate', 'Advanced'
urgency = "High"  # Options: 'High', 'Medium', 'Low'
additional_instructions = "Please ensure to acknowledge the urgency and offer a potential solution."

# Format the messages by replacing the placeholders with actual values
messages = chat_prompt_template_with_few_shots.format_messages(
    email_text=email_text,
    response_length=response_length,
    tone=tone,
    personalization_level=personalization_level,
    language_complexity=language_complexity,
    urgency=urgency,
    additional_instructions=additional_instructions
)

# Print the formatted messages' content only
print('Formatted messages content:')
for msg in messages:
    print(msg)

# Create an instance of the ChatOpenAI model
chat_model: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini-turbo", temperature=0.0)

# Generate the response from the chat model using the messages
chat_model_response = chat_model.invoke(messages)

# Initialize a string output parser to process the model's response
parser = StrOutputParser()

# Extract and format the response from the model's output
formatted_response = parser.invoke(chat_model_response)

# Print the generated response
print(f"Answer: {formatted_response}")