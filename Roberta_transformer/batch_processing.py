import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.eval()

texts = [
    "I love NLP",
    "RoBERTa is powerful",
    "Transformers are useful"
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

print("Input shape:", inputs["input_ids"].shape)

print("Output shape:", outputs.last_hidden_state.shape)