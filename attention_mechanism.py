import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Input sentence
sentence = "The phone battery is not good"

# 2. Tokenization
words = sentence.lower().split()

# Create vocabulary
vocab = {word: i for i, word in enumerate(set(words))}

# Convert words into token IDs
tokens = torch.tensor([[vocab[word] for word in words]])

print("Tokens:", tokens)

# 3. Convert tokens into embeddings
embedding = nn.Embedding(len(vocab), 8)

x = embedding(tokens)

print("\nEmbedding Shape:", x.shape)

# 4. Create Query, Key and Value layers
d_model = 8

Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(x)
K = Wk(x)
V = Wv(x)

# 5. Apply built-in Scaled Dot-Product Attention
output = F.scaled_dot_product_attention(
    Q,
    K,
    V
)

# 6. Display output
print("\nAttention Output:")
print(output)

print("\nOutput Shape:")
print(output.shape)

# 7. Display contextualized word representations
for word, vector in zip(words, output[0]):
    print(f"{word}: {vector.detach().numpy()}")