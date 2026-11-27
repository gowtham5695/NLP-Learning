
# =====================================================
# T5 STEP 4: TEXT GENERATION
# =====================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

MODEL_NAME = "google/flan-t5-small"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

# Input instruction
text = "Explain artificial intelligence in simple words."

# Tokenization
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

# Generate text
with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

# Convert generated IDs into readable text
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Input:")
print(text)

print("\nGenerated Answer:")
print(answer)
