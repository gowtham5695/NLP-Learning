import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
model.eval()

sentences = [
    "I love machine learning",
    "BERT is a powerful model"
]

inputs = tok(
    sentences,
    padding=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1)

masked_embeddings = token_embeddings * mask

sum_embeddings = masked_embeddings.sum(dim=1)

sum_mask = mask.sum(dim=1)

sentence_embeddings = sum_embeddings / sum_mask

print("Token embeddings:", token_embeddings.shape)
print("Sentence embeddings:", sentence_embeddings.shape)
print(sentence_embeddings)