import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model = AutoModelForMaskedLM.from_pretrained("roberta-base")

model.eval()

text = "I love <mask>"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

mask_index = (
    inputs["input_ids"][0] == tokenizer.mask_token_id
).nonzero(as_tuple=True)[0].item()

predicted_id = logits[0, mask_index].argmax()

predicted_token = tokenizer.decode(predicted_id)

print("Predicted token:", predicted_token)