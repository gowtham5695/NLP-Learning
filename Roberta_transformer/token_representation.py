import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.eval()

text = "I love NLP"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

cls_embedding = outputs.last_hidden_state[:, 0, :]

print("Sentence representation:")
print(cls_embedding)

print("Shape:", cls_embedding.shape)