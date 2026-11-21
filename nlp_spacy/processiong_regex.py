import spacy
"""
nlp=spacy.load("en_core_web_sm")

doc=nlp("Hello, world! This is a test sentence for processing with spaCy.")

for t  in doc:
    print(f"{t.text} lemma: {t.lemma_:6} pos: {t.pos_:6} stop: {t.is_stop}")
print([t.text for t in nlp("Don't panic, it's 5 pm.")])

print(nlp.vocab["not"].is_stop)
nlp.vocab["not"].is_stop=False
print(nlp.vocab["not"].is_stop)"""

import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")      # load ONCE, outside the function
nlp.vocab["not"].is_stop = False         # keep negations
nlp.vocab["n't"].is_stop = False

def spacy_text(text):
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
print(spacy_text("The cats were not running and the dogs jumped!"))


raw_docs=[ "The battery life is AMAZING!!!",
    "The battery isn't good",
    "Delivery was late and the battery was bad",
]

cleaned=[" ".join(spacy_text(d)) for d in raw_docs]
print(cleaned)

cv=CountVectorizer()
X=cv.fit_transform(cleaned)
print("feature names: ",cv.get_feature_names_out())
print("vector: ",X.toarray())