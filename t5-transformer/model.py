
# =====================================================
# T5 STEP 1: LOADING A PRETRAINED MODEL
# =====================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# 1. Select pretrained model
MODEL_NAME = "google/flan-t5-large"

# 2. Check available hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Selected device:", device)

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Load pretrained T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 5. Move model to GPU or CPU
model = model.to(device)

# 6. Set evaluation mode
model.eval()

# 7. Count model parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# 8. Display information
print("\n========== MODEL INFORMATION ==========")

print("Model name:", MODEL_NAME)

print("Device:", device)

print("Total parameters:", total_parameters)

print("Tokenizer loaded:", tokenizer is not None)

print("Model loaded:", model is not None)

print("\nT5 model loaded successfully!")
