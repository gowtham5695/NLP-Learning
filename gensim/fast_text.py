from gensim.models import FastText

# Input sentences
sentences = [
    ["nlp", "is", "interesting"],
    ["natural", "language", "processing"],
    ["word", "embeddings", "capture", "meaning"],
    ["fasttext", "learns", "word", "representations"],
    ["machine", "learning", "uses", "text", "data"],
    ["nlp", "helps", "machines", "understand", "language"],
]

# Train FastText model
ft = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=50
)

# Get vector for a word
print(ft.wv["nlp"].shape)

# Get vector for an unseen word
print(ft.wv["nlpp"].shape)