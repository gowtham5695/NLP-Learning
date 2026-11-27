
# =====================================================
# T5 STEP 6: BEAM SEARCH
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

text = """
summarize: Artificial intelligence is a branch of
computer science that enables machines to perform
tasks that normally require human intelligence.
It includes machine learning, deep learning,
natural language processing and computer vision.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=512,
    truncation=True
).to(device)

# Beam search generation
with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

summary = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Original Article:")
print(text)

print("\nBeam Search Summary:")
print(summary)
