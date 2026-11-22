from gensim.models import Word2Vec

sentences = [
    ["i", "love", "nlp"], ["i", "love", "python"],
    ["nlp", "is", "fun"], ["python", "is", "funny"],
]   # a list of tokenized sentences (use your Topic 1 and 2 cleaning)

model = Word2Vec(
    sentences,
    vector_size=100,     # embedding dimension
    window=5,            # context size
    min_count=1,         # ignore words rarer than this (use 5+ on real data)
    sg=1,                # 1 = skip-gram, 0 = CBOW
    negative=5,          # negative samples
    epochs=50,
    workers=4,
)

print(model.wv["nlp"].shape)
print(model.wv.most_similar("nlp", topn=2))
model.save("w2v.model")
model = Word2Vec.load("w2v.model")  