# # ---------USING RAW API
import os
import requests
from dotenv import load_dotenv  

load_dotenv()

# Define the API endpoint and API key
url = "https://api.openai.com/v1/chat/completions"

api_key = os.environ["OPENAI_API_KEY"]

# Define the request headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
# Define the request body
data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Explain how cloud computing works? Assume that I am a 12th class student with no prior technical knowledge"
        }
    ],
    "temperature": 1
}

# Send the request to the OpenAI API
response = requests.post(url, headers=headers, json=data)

# Check if the request was successful and extract the response
if response.ok:
    answer = response.json()['choices'][0]['message']['content']
    print(f"Question: {data['messages'][0]['content']}")
    print(f"Answer: {answer}")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
