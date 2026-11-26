import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model = AutoModel.from_pretrained("roberta-base")

model.eval()

text = "I love NLP"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )

attentions = outputs.attentions

print("Number of layers:", len(attentions))

print("First layer attention shape:", attentions[0].shape)