from typing import Any, Dict, List
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint  # Import pprint for better formatting

# Define a custom callback handler for logging events
class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")
        print(f"Messages: {messages}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print("Response:")  # Label for the response
        pprint(response)  # Use pprint to display the response neatly
        print("Chat model ended, response logged.")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        chain_name = serialized.get('name') if serialized and 'name' in serialized else "Unknown Chain"
        print(f"Chain {chain_name} started")
        print("Serialized Variable:")
        pprint(serialized)  # Use pprint to format the serialized data
        print("Inputs:")
        pprint(inputs)  # Use pprint to format the inputs

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print("Outputs:")  
        pprint(outputs)  # Use pprint to format the outputs
        print("Chain ended")

# Set up chain with GPT-4.0 Mini
callbacks = [LoggingHandler()]  # Attach our custom callback handler
llm = ChatOpenAI(model="gpt-4o-mini")  # Use GPT-4.0 mini model
prompt = ChatPromptTemplate.from_template("What is the square root of {number}?")  # Define the prompt

# Combine the prompt and the model into a chain
chain = prompt | llm | StrOutputParser()

# Run the chain with the callback
response = chain.invoke({"number": "16"}, config={"callbacks": callbacks})

print("Final Response:")
pprint(response)  # Use pprint to display the final response neatly
