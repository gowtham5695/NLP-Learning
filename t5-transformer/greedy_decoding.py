
# =====================================================
# T5 STEP 5: GREEDY DECODING
# =====================================================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

MODEL_NAME = "google/flan-t5-small"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

text = "Explain machine learning in simple words."

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

# Greedy decoding
with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Input:", text)

print("\nGreedy Generated Output:")
print(answer)
