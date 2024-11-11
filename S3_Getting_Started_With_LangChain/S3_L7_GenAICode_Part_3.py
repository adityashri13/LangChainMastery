# # ---------USING OpenAI Python client library

from openai import OpenAI

client = OpenAI()

# Define the prompt (input)
prompt = "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge"

# Function to generate response from OpenAI (model)
def get_completion(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful assistant, skilled in explaining things in simple and easy to understand manner."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion

# Function to extract proper response from the completion
def extract_response(completion):
    return completion.choices[0].message.content

# Generate and extract responses for all prompts (output)
response = get_completion(prompt)

# Print the responses
print(f"Question: {prompt}")
print(f"Answer: {extract_response(response)}")
