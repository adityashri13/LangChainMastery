import os
from together import Together
from dotenv import load_dotenv

load_dotenv()

# Initialize the Together API client with the API key from environment variables
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

# Get query from the user
user_query = input("Please enter your query: ")

# Create the chat completion with the user-provided query
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {
            "role": "user",
            "content": user_query  # Use the user-provided query here
        }
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["[/INST]", "</s>"],
    stream=False
)

# Extract and print the content of the assistant's response
if response and response.choices:
    print("Assistant Response:")
    print(response.choices[0].message.content)
else:
    print("No response from the model.")