
from transformers import pipeline

# Load pretrained sentiment analysis model
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentences = [
    "This movie is amazing!",
    "I hate this product.",
    "The food was excellent.",
    "The service was terrible."
]

# Predict sentiment
results = sentiment(sentences)

for sentence, result in zip(sentences, results):
    print("Sentence:", sentence)
    print("Sentiment:", result["label"])
    print("Confidence:", result["score"])
    print("-----------------------")
