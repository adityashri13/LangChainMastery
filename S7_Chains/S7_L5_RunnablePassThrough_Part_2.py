from langchain_core.runnables import RunnablePassthrough

# Step 1: Convert sentences to uppercase
def convert_to_uppercase(sentences):
    return [sentence.upper() for sentence in sentences]

# Step 2: Filter out sentences containing certain words
def filter_sentences(sentences, forbidden_words):
    return [sentence for sentence in sentences if not any(word in sentence for word in forbidden_words)]

# Step 3: Count the number of words in each sentence
def count_words(sentences):
    return [len(sentence.split()) for sentence in sentences]

# Step 4: Combine the filtering function with the forbidden words
def filter_out_forbidden_words(sentences):
    forbidden_words = {"SECRET", "SUNNY"}
    return filter_sentences(sentences, forbidden_words)

# Input data
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world!",
    "Python is an amazing programming language.",
    "The weather is sunny today.",
    "This is a secret message."
]

# Create a passthrough (not strictly needed here, but included to illustrate chaining)
input_transformer = RunnablePassthrough()

# Set up the chain of operations
chain = (
    input_transformer  # Passes data through unchanged
    | convert_to_uppercase  # Step 1: Convert to uppercase
    | filter_out_forbidden_words  # Step 2: Filter sentences
    | count_words  # Step 3: Count words in each sentence
)

# Invoke the chain
result = chain.invoke(sentences)

# Print the results
print("Word counts after processing:", result)  
