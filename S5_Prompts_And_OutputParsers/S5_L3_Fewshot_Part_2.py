from langchain_core.prompts import (ChatPromptTemplate, 
 FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI

examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{input}"),
        SystemMessagePromptTemplate.from_template("{output}")
    ]
)

print("-------example_prompt")
print(example_prompt)
print(type(example_prompt))

few_shot_example_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"]
)

print("-------few_shot_example_template")
print(few_shot_example_template)
print(type(few_shot_example_template))

chat_prompt_template_with_few_shots = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a wondrous wizard of math."),
        few_shot_example_template,
        HumanMessagePromptTemplate.from_template("{user_query}"),
    ]
)

print("-------chat_prompt_template_with_few_shots")
print(chat_prompt_template_with_few_shots)
print(type(chat_prompt_template_with_few_shots))


messages = chat_prompt_template_with_few_shots.format_messages(user_query="What is 2 🦜 9?")

print("-------After formatting: messages")
print(messages)
print(type(messages))

model = ChatOpenAI(model="gpt-4o-mini")

response = model.invoke(messages)

print(response)