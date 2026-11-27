
# =====================================================
# T5 STEP 7: SAMPLING
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

text = "Write a short creative story about a robot."

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

# Generate three different samples
for i in range(3):

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.90,
            temperature=0.8
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(f"\nGeneration {i + 1}:")
    print(answer)

    print("-" * 60)
