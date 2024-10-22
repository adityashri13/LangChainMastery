# Import necessary classes from LangChain and OpenAI
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import re

# Step 1: Data Cleaning Function
def clean_text(text):
    # Remove special characters and normalize whitespace
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

# Step 2: Text Transformation Function
def transform_text(cleaned_text):
    # Convert text to uppercase for standardization
    return cleaned_text.upper()

# Step 3: Sentiment Analysis using ChatOpenAI
def openai_sentiment_analysis(text):
    # Initialize ChatOpenAI model
    chat_model = ChatOpenAI(model_name="gpt-4o-mini")
    # Create prompt for sentiment analysis
    prompt = f"Analyze the sentiment of the following text and classify it as positive, negative, or neutral:\n{text}"
    # Get the response from OpenAI model
    response = chat_model.invoke(prompt)
    # Extract the text from the AIMessage object
    sentiment = response.content.strip().lower()
    return {'text': text, 'sentiment': sentiment}

# Step 4: Summary Aggregation Function
def aggregate_sentiments(sentiment_results):
    # Count occurrences of each sentiment type
    summary = {'positive': 0, 'negative': 0, 'neutral': 0}
    for result in sentiment_results:
        if result['sentiment'] in summary:
            summary[result['sentiment']] += 1
    return summary

# Wrap each function in RunnableLambda
runnable_clean = RunnableLambda(lambda texts: [clean_text(text) for text in texts])
runnable_transform = RunnableLambda(lambda cleaned_texts: [transform_text(text) for text in cleaned_texts])
runnable_sentiment = RunnableLambda(lambda transformed_texts: [openai_sentiment_analysis(text) for text in transformed_texts])
runnable_aggregate = RunnableLambda(aggregate_sentiments)

# Create a sequence of operations using RunnableSequence
review_analysis_pipeline = RunnableSequence(
    first=runnable_clean,                   # Clean text
    middle=[runnable_transform, runnable_sentiment],  # Transform and analyze sentiment
    last=runnable_aggregate                 # Aggregate sentiment scores
)

# Sample input data: a batch of customer reviews
reviews = [
    "The product quality is excellent and the service was awesome!",
    "Very poor experience, the product was bad and delivery was horrible.",
    "Good value for money, but could be better.",
    "I had a bad experience, but the customer support was good.",
    "This is the best product I've ever bought!"
]

# Execute the pipeline with the batch of reviews
result = review_analysis_pipeline.invoke(reviews)

# Output the final aggregated sentiment result
print(result)  # Expected output might vary based on OpenAI responses, e.g., {'positive': 3, 'negative': 2, 'neutral': 0}