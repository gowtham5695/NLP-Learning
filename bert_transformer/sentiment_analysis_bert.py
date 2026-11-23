from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="textattack/bert-base-uncased-SST-2"
)

sentences = [
    "This movie is amazing!",
    "I hate this product.",
    "The food was excellent.",
    "The service was terrible."
]

results = sentiment(sentences)

for sentence, result in zip(sentences, results):
    print("Sentence:", sentence)
    print("Sentiment:", result["label"])
    print("Confidence:", result["score"])
    print("-------------------")