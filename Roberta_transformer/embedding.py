import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.eval()

text = "I love NLP"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    embeddings = model.embeddings.word_embeddings(
        inputs["input_ids"]
    )

print("Word embeddings:", embeddings)
print("Shape:", embeddings.shape)