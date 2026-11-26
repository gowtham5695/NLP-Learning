from transformers import AutoModel, AutoTokenizer

name="roberta-base"

tok=AutoTokenizer.from_pretrained(name)
model=AutoModel.from_pretrained(name)

model.eval()
t="I love NLP"
token=tok.tokenize(t)
print(token)



input_ids = tok.encode(t)

print(input_ids)


encoded = tok(t)

print("Input IDs:", encoded["input_ids"])

print(
    "Tokens:",
    tok.convert_ids_to_tokens(encoded["input_ids"])
)



inputs = tok(t, return_tensors="pt")

print("Input IDs:", inputs["input_ids"])

print("Attention Mask:", inputs["attention_mask"])