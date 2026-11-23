import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
model.eval()

sentences = [
    "I love NLP",
    "BERT is powerful",
    "I am learning artificial intelligence"
]

inputs = tok(
    sentences,
    padding=True,
    return_tensors="pt"
)

print("Input IDs:")
print(inputs["input_ids"])

print("Attention Mask:")
print(inputs["attention_mask"])

for i in range(len(sentences)):
    print(tok.convert_ids_to_tokens(inputs["input_ids"][i]))

with torch.no_grad():
    outputs = model(**inputs)

print("Output shape:", outputs.last_hidden_state.shape)