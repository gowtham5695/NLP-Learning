from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

model.save_pretrained("my_roberta")
tokenizer.save_pretrained("my_roberta")

print("Model saved successfully!")

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("my_roberta")

model = AutoModel.from_pretrained("my_roberta")

model.eval()

print("Model reloaded successfully!")