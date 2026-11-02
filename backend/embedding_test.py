from embedding import create_embeddings

texts = [
    "Supervised learning uses labeled training data.",
    "Machine learning models learn from examples.",
    "Pizza is a popular food."
]


embeddings = create_embeddings(
    texts
)


print(
    "Number of texts:",
    len(texts)
)


print(
    "Embedding shape:",
    embeddings.shape
)


print(
    "\nFirst embedding:"
)


print(
    embeddings[0]
)