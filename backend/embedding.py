from sentence_transformers import SentenceTransformer


# Load the embedding model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(text_chunks):
    """
    Convert multiple text chunks into normalized embeddings.

    Parameters:
        text_chunks: list of text strings

    Returns:
        numpy array containing embeddings
    """

    embeddings = model.encode(
        text_chunks,
        normalize_embeddings=True
    )

    return embeddings


def create_query_embedding(query):
    """
    Convert a user's question into an embedding.

    Parameters:
        query: user's question as a string

    Returns:
        numpy array containing the query embedding
    """

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return embedding