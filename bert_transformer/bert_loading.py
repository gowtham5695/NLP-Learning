from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

model.eval()

print(model)