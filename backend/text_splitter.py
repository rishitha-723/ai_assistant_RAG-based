def split_text(text, chunk_size=100, overlap=20):
    """
    Split text into smaller chunks with overlapping words.

    chunk_size: maximum number of words in each chunk.
    overlap: number of words shared between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks