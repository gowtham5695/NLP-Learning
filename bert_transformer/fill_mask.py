from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

sentence = "I love learning [MASK]."

results = fill_mask(sentence, top_k=5)

for result in results:
    print(
        result["token_str"],
        "->",
        result["score"]
    )