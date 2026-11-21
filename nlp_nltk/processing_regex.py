"""import nltk
nltk.download("punckt")
nltk.download("punkt_tab")      # needed by newer NLTK versions
nltk.download("stopwords")
nltk.download("wordnet")        # dictionary used by the lemmatizer

from nltk.tokenize import word_tokenize

print(word_tokenize("Don't panic, it's fine."))

from nltk.corpus import stopwords

stop=set(stopwords.words("english"))
print(len(stop))

print("not" in stop, "no" in stop, "the" in stop)

stop=stop-{"not","no","the"}
print("not" in stop, "no" in stop, "the" in stop)

from nltk.stem import PorterStemmer

st=PorterStemmer()
for i in ["running","ran","runs","easily","fairly"]:
    print(f"{i} -> {st.stem(i)}")

from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()
for i in ["running","ran","runs","easily","fairly","better","best"]:
    print(f"{i} -> {lem.lemmatize(i,pos='v'),lem.lemmatize(i,pos='a')}")
"""

#complete NLTK pipeline

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
ps = PorterStemmer()

def process_text(text):
    text = re.sub(r"<.*?>|https?://\S+", " ", text.lower())   # HTML + URLs
    tokens = word_tokenize(text)
    tokens = ["not" if t == "n't" else t for t in tokens]     # keep negation
    tokens = [t for t in tokens if t.isalpha() and t not in stop]
    return [ps.stem(t) for t in tokens]

# test your sentence
print(process_text("This is a test. Don't panic, it's fine. Visit https://example.com for more info."))

# cleaning -> CountVectorizer
raw_docs = [
    "The battery life is AMAZING!!!",
    "The battery isn't good",
    "Delivery was late and the battery was bad",
]

cleaned = [" ".join(process_text(d)) for d in raw_docs]
print(cleaned)

cv = CountVectorizer()
X = cv.fit_transform(cleaned)
print(cv.get_feature_names_out())
print(X.toarray())
