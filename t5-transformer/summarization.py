
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

article = """
Artificial Intelligence is changing the way people work and live.
Machine Learning is a branch of Artificial Intelligence that allows
computers to learn patterns from data.

Deep Learning is a type of Machine Learning that uses neural networks
with multiple layers. It is widely used in image recognition,
speech recognition and natural language processing.

Today, companies use AI in healthcare, education, banking and
transportation to automate tasks and improve decision making.
"""

prompt = """
Write a summary of the following article.
Include all its major ideas.
Use exactly three short sentences.

Article:
""" + article

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        min_new_tokens=35,
        num_beams=4,
        no_repeat_ngram_size=2
    )

print("GENERATED SUMMARY:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
