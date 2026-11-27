
# =====================================================
# T5 STEP 1: PRETRAINED MODEL + TOKENIZATION
# =====================================================

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Model name
MODEL_NAME = "google/flan-t5-small"

# 2. Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Load pretrained T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 5. Move model to device
model = model.to(device)

# 6. Evaluation mode
model.eval()

print("Model loaded successfully!")
print("Device:", device)
print("Model:", MODEL_NAME)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# =====================================================
# TOKENIZATION
# =====================================================

text = "Artificial intelligence is changing the world."

# Convert text into tokens
tokens = tokenizer.tokenize(text)

# Convert text into token IDs
inputs = tokenizer(text, return_tensors="pt")

print("\nOriginal Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nInput IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

# Convert IDs back into text
decoded = tokenizer.decode(
    inputs["input_ids"][0],
    skip_special_tokens=True
)

print("\nDecoded Text:")
print(decoded)


