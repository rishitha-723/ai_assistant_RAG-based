
import faiss
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# CREATE FAISS INDEX
# ==========================================

def create_vector_index(embeddings):
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must contain at least one vector.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# ==========================================
# ADD EMBEDDINGS TO EXISTING INDEX
# ==========================================

def add_embeddings_to_index(index, embeddings):
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return index

    index.add(embeddings)

    return index


# ==========================================
# EXISTING SEMANTIC SEARCH
# ==========================================

def search_vector_index(index, query_embedding, top_k=5):
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    if index is None or index.ntotal == 0:
        return np.array([]), np.array([], dtype="int64")

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]


# ==========================================
# KEYWORD SEARCH USING TF-IDF
# ==========================================

def search_keyword_scores(query, text_chunks):
    """
    Calculate keyword relevance using TF-IDF.

    TF-IDF gives importance to words based on how
    frequently they occur in a chunk and how rare
    they are across the collection.
    """

    if not text_chunks:
        return np.array([], dtype="float32")

    clean_chunks = [
        text if isinstance(text, str) and text.strip()
        else "empty"
        for text in text_chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w[\w+#.-]*\b"
    )

    try:
        chunk_vectors = vectorizer.fit_transform(clean_chunks)
        query_vector = vectorizer.transform([query])

        keyword_scores = (
            chunk_vectors @ query_vector.T
        ).toarray().flatten()

        return keyword_scores.astype("float32")

    except ValueError:
        # Handles cases where the text contains no usable terms.
        return np.zeros(
            len(text_chunks),
            dtype="float32"
        )


# ==========================================
# HYBRID SEARCH
# ==========================================

def search_hybrid(
    index,
    query_embedding,
    query,
    text_chunks,
    top_k=5,
    semantic_weight=0.65,
    keyword_weight=0.35
):
    """
    Combine semantic similarity and keyword relevance.

    semantic_weight: importance of semantic search.
    keyword_weight: importance of keyword search.

    Returns ranked results containing:
    - chunk index
    - semantic score
    - keyword score
    - combined score
    """

    if not text_chunks:
        return []

    if index is None or index.ntotal == 0:
        return []

    if index.ntotal != len(text_chunks):
        raise ValueError(
            "FAISS index and document chunks are not aligned."
        )

    if not np.isclose(
        semantic_weight + keyword_weight,
        1.0
    ):
        raise ValueError(
            "Semantic and keyword weights must add up to 1."
        )

    # Search all chunks semantically so keyword matches
    # are not excluded before the two methods are combined.
    semantic_scores, semantic_indices = search_vector_index(
        index,
        query_embedding,
        top_k=index.ntotal
    )

    # Calculate keyword relevance for every chunk.
    keyword_scores = search_keyword_scores(
        query,
        text_chunks
    )

    results = []

    for semantic_score, chunk_index in zip(
        semantic_scores,
        semantic_indices
    ):
        chunk_index = int(chunk_index)

        if chunk_index < 0 or chunk_index >= len(text_chunks):
            continue

        # Embedding cosine similarity can be negative.
        # Clamp it to [0, 1] before combining.
        semantic_score = float(
            np.clip(semantic_score, 0.0, 1.0)
        )

        keyword_score = float(
            keyword_scores[chunk_index]
        )

        combined_score = (
            semantic_weight * semantic_score
            + keyword_weight * keyword_score
        )

        results.append({
            "index": chunk_index,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "combined_score": combined_score
        })

    results.sort(
        key=lambda item: item["combined_score"],
        reverse=True
    )

    return results[:min(top_k, len(results))]
