# Import necessary classes from the LangChain library
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnableLambda

# Define basic functions for arithmetic operations
def add(x):
    return x + 2

def multiply(x):
    return x * 3

def subtract(x):
    return x - 1

# Create RunnableLambda instances for each function
runnable_add = RunnableLambda(add)       # Wrap the 'add' function as a Runnable
runnable_mul = RunnableLambda(multiply)  # Wrap the 'multiply' function as a Runnable
runnable_sub = RunnableLambda(subtract)  # Wrap the 'subtract' function as a Runnable

# Define a sequence of operations using RunnableSequence
sequence = RunnableSequence(
    first=runnable_add,                  # The first step is 'runnable_add'
    middle=[runnable_mul],               # The middle step is 'runnable_mul'
    last=runnable_sub                    # The last step is 'runnable_sub'
)

# Alternatively, you can chain the Runnables directly (if supported)
# sequence = runnable_add | runnable_mul | runnable_sub

# Execute the sequence with an initial input of 5
result = sequence.invoke(5)  # Apply (5 + 2) -> 7, then (7 * 3) -> 21, then (21 - 1) -> 20
print(result)  # Output the final result of the sequence, which is 20