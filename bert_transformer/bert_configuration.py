
from transformers import AutoConfig, AutoModel

name = "bert-base-uncased"

# Load BERT configuration
config = AutoConfig.from_pretrained(name)

print("Number of layers:", config.num_hidden_layers)
print("Hidden size:", config.hidden_size)
print("Attention heads:", config.num_attention_heads)
print("Vocabulary size:", config.vocab_size)
print("Maximum position embeddings:", config.max_position_embeddings)

# Load BERT model
model = AutoModel.from_pretrained(name)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_params)
print("Total parameters in millions:", total_params / 1_000_000)
