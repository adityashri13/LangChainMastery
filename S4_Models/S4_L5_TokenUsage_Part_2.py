from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

OPENAI_PRICING = {
    'gpt-4o': {'input_cost_per_million_tokens': 5.00, 'output_cost_per_million_tokens': 15.00},
    'gpt-4o-2024-05-13': {'input_cost_per_million_tokens': 5.00, 'output_cost_per_million_tokens': 15.00},
    'gpt-4o-2024-08-06': {'input_cost_per_million_tokens': 2.50, 'output_cost_per_million_tokens': 10.0},
}

GOOGLE_PRICING = {
    'gemini-1.5-flash': {
        'input_cost_per_1k_chars': {'up_to_128k': 0.00001875, 'over_128k': 0.0000375},
        'output_cost_per_1k_chars': {'up_to_128k': 0.000075, 'over_128k': 0.00015},
    },
    'gemini-1.5-pro': {
        'input_cost_per_1k_chars': {'up_to_128k': 0.00125, 'over_128k': 0.0025},
        'output_cost_per_1k_chars': {'up_to_128k': 0.00375, 'over_128k': 0.0075},
    }
}

def count_characters(text):
    return len(text)

def calculate_prompt_length(input_text, output_text):
    return len(input_text) + len(output_text)

def get_openai_cost(input_tokens, output_tokens, pricing):
    input_cost = (input_tokens / 1_000_000) * pricing['input_cost_per_million_tokens']
    output_cost = (output_tokens / 1_000_000) * pricing['output_cost_per_million_tokens']
    return input_cost + output_cost

def get_google_cost(input_chars, output_chars, pricing, prompt_length):
    if prompt_length <= 128_000:
        input_cost = (input_chars / 1_000) * pricing['input_cost_per_1k_chars']['up_to_128k']
        output_cost = (output_chars / 1_000) * pricing['output_cost_per_1k_chars']['up_to_128k']
    else:
        input_cost = (input_chars / 1_000) * pricing['input_cost_per_1k_chars']['over_128k']
        output_cost = (output_chars / 1_000) * pricing['output_cost_per_1k_chars']['over_128k']
    return input_cost + output_cost

def calculate_llm_cost(model, input_chars, output_chars, prompt_length=None):
    if model in OPENAI_PRICING:
        input_tokens = input_chars // 4
        output_tokens = output_chars // 4
        return get_openai_cost(input_tokens, output_tokens, OPENAI_PRICING[model])
    elif model in GOOGLE_PRICING:
        if prompt_length is None:
            raise ValueError("Prompt length is required for Google models.")
        return get_google_cost(input_chars, output_chars, GOOGLE_PRICING[model], prompt_length)
    else:
        raise ValueError("Unsupported model.")



user_query = "Explain how cloud computing works? Assume that I am 12th class student with no prior technical knowledge"

messages = [
    SystemMessage(content="You are a helpful assistant, skilled in explaining things in a simple and easy to understand manner."),
    HumanMessage(content=user_query)
]

model = ChatOpenAI(model="gpt-4o")

response = model.invoke(messages)

parser = StrOutputParser()
formatted_response = parser.invoke(response)
print(formatted_response)


input_chars = count_characters(user_query)
output_chars = count_characters(formatted_response)
prompt_length = calculate_prompt_length(user_query, formatted_response)

model = "gpt-4o"
cost = calculate_llm_cost(model, input_chars, output_chars, prompt_length=prompt_length)
print(f"The estimated cost for this OpenAI GPT 4o API call is: ${cost}")
