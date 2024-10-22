class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # the other func consumes the result of this func
            return other(self.func(*args, **kwargs))
        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# Define standalone functions
def add(x):
    return x + 2

def multiply(x):
    return x * 3

# Create Runnable instances for each function
add_runnable = Runnable(add)
multiply_runnable = Runnable(multiply)

# Chain the Runnable objects
pipeline = add_runnable | multiply_runnable

# Execute the pipeline
result = pipeline(5)  # First adds 2, then multiplies by 3
print(result)  # Output: 21
