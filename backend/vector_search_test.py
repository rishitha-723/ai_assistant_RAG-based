from embedding import (
    create_embeddings,
    create_query_embedding
)

from vector_store import (
    create_vector_index,
    search_vector_index
)


# Sample document chunks
texts = [
    "Supervised learning uses labeled training data.",
    
    "Deep learning uses neural networks with multiple layers.",
    
    "Pizza is a popular food enjoyed by many people.",
    
    "Unsupervised learning finds patterns in data without labels."
]


print("Creating embeddings...")

embeddings = create_embeddings(
    texts
)

print(
    "Embedding shape:",
    embeddings.shape
)


print("\nCreating FAISS index...")

index = create_vector_index(
    embeddings
)

print(
    "Number of vectors in index:",
    index.ntotal
)


# User's question
query = "What is supervised machine learning?"


print(
    "\nUser question:",
    query
)


print("\nCreating query embedding...")

query_embedding = create_query_embedding(
    query
)


print("\nSearching FAISS index...")

scores, indices = search_vector_index(
    index,
    query_embedding,
    top_k=3
)


print("\nTop results:\n")


for rank, (score, index_number) in enumerate(
    zip(scores, indices),
    start=1
):

    print(
        f"Rank {rank}"
    )

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Text: {texts[index_number]}"
    )

    print()