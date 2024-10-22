from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini-0125", temperature=0.0)

response = model.invoke("What is 2 🦜 9?")

print(response)