from transformers import pipeline
import time

text = "This movie is amazing!"

# BERT
start = time.time()

bert = pipeline(
    "sentiment-analysis",
    model="textattack/bert-base-uncased-SST-2"
)

bert_result = bert(text)
bert_time = time.time() - start

# DistilBERT
start = time.time()

distilbert = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

distilbert_result = distilbert(text)
distilbert_time = time.time() - start

print("BERT:", bert_result)
print("BERT time:", bert_time)

print("DistilBERT:", distilbert_result)
print("DistilBERT time:", distilbert_time)