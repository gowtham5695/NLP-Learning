import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.eval()

text = "I love NLP"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state

print("Last hidden state:")
print(last_hidden_state)

print("Shape:", last_hidden_state.shape)