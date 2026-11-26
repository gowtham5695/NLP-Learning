import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.eval()

sentence1 = "I love NLP"

sentence2 = "Transformers are powerful"

inputs = tokenizer(
    sentence1,
    sentence2,
    return_tensors="pt"
)

print("Input IDs:", inputs["input_ids"])

print(
    "Tokens:",
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
)

with torch.no_grad():
    outputs = model(**inputs)

print("Output shape:", outputs.last_hidden_state.shape)