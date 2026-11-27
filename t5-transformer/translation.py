
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

sentences = [
    "Translate English to German: Good morning.",
    "Translate English to French: How are you?",
    "Translate English to Spanish: I love artificial intelligence."
]

for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nInput:", sentence)
    print("Output:", answer)
    print("-" * 50)
