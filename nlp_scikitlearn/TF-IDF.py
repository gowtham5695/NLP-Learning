from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "battery is good",
    "battery is bad",
    "phone is excellent",
    "phone is terrible",
    "the camera is amazing",
    "camera quality is poor",
    "the phone works great",
    "the phone is not good",
    "very happy with this product",
    "very disappointed with this product",
]

# Create TF-IDF vectorizer
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

# Learn vocabulary and calculate TF-IDF values
X = tv.fit_transform(texts)

# Display vocabulary
print("Vocabulary:")
print(tv.get_feature_names_out())

# Display TF-IDF matrix
print("\nTF-IDF Matrix:")
print(X.toarray())