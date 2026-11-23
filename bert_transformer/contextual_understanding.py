import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
model.eval()

sentences = [
    "I deposited money in the bank",
    "I sat near the river bank"
]

inputs = tok(
    sentences,
    padding=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

# Extract the bank token representations
bank1 = outputs.last_hidden_state[0, 6, :]
bank2 = outputs.last_hidden_state[1, 6, :]

similarity = F.cosine_similarity(
    bank1.unsqueeze(0),
    bank2.unsqueeze(0)
)

print("Bank vector 1:", bank1.shape)
print("Bank vector 2:", bank2.shape)
print("Cosine similarity:", similarity.item())