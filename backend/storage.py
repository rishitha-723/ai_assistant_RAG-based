import json
from pathlib import Path

import faiss


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Persistent storage directory
DATA_DIR = BASE_DIR / "data"

# Files used for persistence
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"


def ensure_data_directory():
    """
    Create the data directory if it does not exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_vector_store(index, document_chunks, document_names):
    """
    Save the FAISS index and document metadata to disk.
    """

    ensure_data_directory()

    # Save FAISS vector index
    if index is not None:
        faiss.write_index(index, str(INDEX_PATH))

    # Save metadata
    metadata = {
        "document_chunks": document_chunks,
        "document_names": document_names
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_vector_store():
    """
    Load the FAISS index and metadata from disk.

    Returns:
        index,
        document_chunks,
        document_names
    """

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, [], []

    try:
        # Load FAISS index
        index = faiss.read_index(str(INDEX_PATH))

        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        document_chunks = metadata.get("document_chunks", [])
        document_names = metadata.get("document_names", [])

        return index, document_chunks, document_names

    except Exception:
        print("Warning: Could not load saved document data.")
        return None, [], []


def clear_vector_store():
    """
    Delete the saved FAISS index and metadata.
    """

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    if METADATA_PATH.exists():
        METADATA_PATH.unlink()