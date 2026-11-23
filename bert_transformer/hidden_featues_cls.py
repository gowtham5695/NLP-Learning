import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
model.eval()

sentence = "I love learning NLP"

inputs = tok(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state

print("Hidden states shape:", hidden_states.shape)

# Extract CLS representation
cls_vector = hidden_states[:, 0, :]

print("CLS vector shape:", cls_vector.shape)

# Pooler output
print("Pooler output shape:", outputs.pooler_output.shape)