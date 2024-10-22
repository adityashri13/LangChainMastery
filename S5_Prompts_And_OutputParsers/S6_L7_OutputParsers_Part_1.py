from pydantic import BaseModel, ValidationError

# Define a data model
class User(BaseModel):
    name: str
    age: int
    email: str

# Validate and parse data
try:
    user = User(name="Alice", age=30, email="alice@example.com")
    print(user)
except ValidationError as e:
    print(e)

# Invalid data
try:
    user = User(name="Alice", age="thirty", email="alice@example.com")
except ValidationError as e:
    print(e)
