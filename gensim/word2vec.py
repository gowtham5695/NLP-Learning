#Load pretrained vectors and use them (most common in real work)
import gensim.downloader as api

wv=api.load("glove-wiki-gigaword-100")

print(wv["king"].shape)
print(wv.similarity("good","great"))
print(wv.similarity("king","table"))
print(wv.most_similar("king",topn=3))
print(wv.most_similar(positive=["king","queen"],negative=["man"],topn=1))
print("king" in wv)