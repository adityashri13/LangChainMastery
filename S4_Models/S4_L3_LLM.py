from langchain_openai import OpenAI

# -------INPUT
# Define the user query (prompt/input)
user_query = "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge"


# -------MODEL
# Initialize the OpenAI model using LangChain (model)
llm = OpenAI()


# -------OUTPUT
# Generate the response from the model (model)
model_response = llm.invoke(user_query)

# Print the question and the generated response (output)
print(f"Question: {user_query}")
print(f"Answer Type: {type(model_response)}")
print(f"Answer: {model_response}")

