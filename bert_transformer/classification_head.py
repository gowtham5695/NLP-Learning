import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "textattack/bert-base-uncased-SST-2"

tok = AutoTokenizer.from_pretrained(name)

model = AutoModelForSequenceClassification.from_pretrained(name)

model.eval()

sentence = "I really enjoyed this movie!"

inputs = tok(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

predicted_class = torch.argmax(logits, dim=1).item()

label = model.config.id2label[predicted_class]

print("Logits:", logits)
print("Predicted class:", predicted_class)
print("Sentiment:", label)