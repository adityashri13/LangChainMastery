from langchain_core.runnables import RunnablePassthrough

# Let's say we have a list of messages
messages = ["Hello", "How are you?", "Goodbye"]

# We create a passthrough
input_transformer = RunnablePassthrough()

# Then we use it in a chain of operations
def convert_to_uppercase(messages):
    return [msg.upper() for msg in messages]

chain = input_transformer | convert_to_uppercase

# ALTERNATE APPROACH USING LAMBDA
# chain = input_transformer | (lambda messages: [msg.upper() for msg in messages])

# When we invoke the chain, the messages are converted to uppercase
result = chain.invoke(messages)
print(result)  # Output: ['HELLO', 'HOW ARE YOU?', 'GOODBYE']