from transformers import AutoTokenizer, AutoModel

name="roberta-base"

tok=AutoTokenizer.from_pretrained(name)
model=AutoModel.from_pretrained(name)

model.eval()

print("model loaded successfully")