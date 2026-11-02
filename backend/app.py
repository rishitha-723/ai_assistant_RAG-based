from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles

from document_processor import process_pdf
from embedding import create_embeddings, create_query_embedding
from vector_store import create_vector_index, search_hybrid
from llm import generate_answer
from storage import load_vector_store, save_vector_store, clear_vector_store


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

app = FastAPI(
    title="StudyAI",
    description="AI-powered college document assistant",
    version="2.2",
)

app.mount(
    "/frontend",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)

vector_index, document_chunks, document_names = load_vector_store()

TOP_K = 8
MAX_CONTEXT_CHUNKS = 4
MIN_COMBINED_SCORE = 0.18


@app.get("/")
def home():
    return {"message": "StudyAI backend is running!"}


@app.get("/documents")
def get_documents():
    return {
        "documents": document_names,
        "total_documents": len(document_names),
        "total_chunks": len(document_chunks),
        "vectors_in_index": vector_index.ntotal if vector_index is not None else 0,
    }


def is_summary_question(question):
    """Identify common requests for a document summary."""
    summary_phrases = (
        "summarize",
        "summarise",
        "summary",
        "overview",
        "main topics",
        "key points",
        "important topics",
        "main points",
    )
    question_lower = question.lower()
    return any(phrase in question_lower for phrase in summary_phrases)


def select_summary_chunks(chunks, max_chunks=20):
    """Select chunks distributed across the document or collection."""
    if len(chunks) <= max_chunks:
        return chunks

    if max_chunks <= 1:
        return [chunks[0]]

    selected = []
    step = (len(chunks) - 1) / (max_chunks - 1)

    for position in range(max_chunks):
        index = round(position * step)
        selected.append(chunks[index])

    return selected


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_index, document_chunks, document_names

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file."}

    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        return {"error": "Invalid file type. Please upload a PDF."}

    temp_path = BASE_DIR / f"uploaded_{uuid.uuid4().hex}.pdf"

    try:
        with open(temp_path, "wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        new_chunks, total_pages = process_pdf(str(temp_path), file.filename)

        if not new_chunks:
            return {
                "error": (
                    "No readable text was found in this PDF. "
                    "It may be scanned or image-only."
                )
            }

        # Replace an existing document with the same filename.
        remaining_chunks = [
            chunk for chunk in document_chunks
            if chunk.get("source") != file.filename
        ]
        combined_chunks = remaining_chunks + new_chunks

        all_texts = [chunk["text"] for chunk in combined_chunks]
        all_embeddings = create_embeddings(all_texts)
        new_index = create_vector_index(all_embeddings)

        updated_names = [
            name for name in document_names if name != file.filename
        ]
        updated_names.append(file.filename)

        document_chunks = combined_chunks
        document_names = updated_names
        vector_index = new_index

        save_vector_store(vector_index, document_chunks, document_names)

        return {
            "message": "Document uploaded successfully.",
            "filename": file.filename,
            "pages": total_pages,
            "total_documents": len(document_names),
            "total_chunks": len(document_chunks),
        }

    except Exception as exc:
        print(f"PDF upload error: {exc}")
        return {
            "error": (
                "Could not process this PDF. "
                "Check the backend terminal for details."
            )
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()
        await file.close()


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    document_name: str = Form(""),
):
    if not question.strip():
        return {"error": "Please enter a question."}

    if not document_chunks:
        return {"error": "Please upload a PDF first."}

    selected_document = document_name.strip()

    if selected_document:
        if selected_document not in document_names:
            return {"error": "The selected document is not loaded."}

        searchable_chunks = [
            chunk for chunk in document_chunks
            if chunk.get("source") == selected_document
        ]
    else:
        searchable_chunks = document_chunks

    if not searchable_chunks:
        return {
            "question": question,
            "answer": "No readable content was found in this document.",
            "sources": [],
        }

    # Summary requests use chunks distributed across the selected collection.
    # Other questions use hybrid semantic + keyword retrieval.
    if is_summary_question(question):
        relevant_chunks = select_summary_chunks(
            searchable_chunks,
            max_chunks=20,
        )
    else:
        searchable_texts = [
            chunk.get("text", "") for chunk in searchable_chunks
        ]

        if selected_document:
            # Build a temporary index so results can only come from this PDF.
            selected_embeddings = create_embeddings(searchable_texts)
            search_index = create_vector_index(selected_embeddings)
        else:
            search_index = vector_index

        if search_index is None:
            return {"error": "No searchable documents are available."}

        query_embedding = create_query_embedding(question)

        # The returned indexes refer to searchable_chunks in the same order
        # as searchable_texts and the index used for this search.
        ranked_results = search_hybrid(
            index=search_index,
            query_embedding=query_embedding,
            query=question,
            text_chunks=searchable_texts,
            top_k=min(TOP_K, len(searchable_chunks)),
        )

        relevant = [
            item for item in ranked_results
            if item["combined_score"] >= MIN_COMBINED_SCORE
        ][:MAX_CONTEXT_CHUNKS]

        relevant_chunks = [
            searchable_chunks[item["index"]]
            for item in relevant
        ]

    if not relevant_chunks:
        return {
            "question": question,
            "answer": (
                "I couldn't find a clear answer in the uploaded "
                "document. Try rephrasing your question."
            ),
            "sources": [],
        }

    context_parts = []
    sources = []

    for chunk in relevant_chunks:
        source_name = chunk.get("source", "Unknown document")
        page_number = chunk.get("page", "Unknown")
        heading = chunk.get("heading", "Unknown section")

        context_parts.append(
            f"Document: {source_name}\n"
            f"Page: {page_number}\n"
            f"Section: {heading}\n"
            f"Content:\n{chunk.get('text', '')}"
        )

        source = {"source": source_name, "page": page_number}
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    try:
        answer = generate_answer(question, context)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    except Exception as exc:
        print(f"Gemini generation error: {exc}")
        return {
            "error": (
                "StudyAI could not generate an answer. "
                "Check your API key, internet connection, "
                "and backend terminal."
            )
        }


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    global vector_index, document_chunks, document_names

    if document_name not in document_names:
        return {"error": "Document not found."}

    remaining_chunks = [
        chunk for chunk in document_chunks
        if chunk.get("source") != document_name
    ]
    remaining_names = [
        name for name in document_names if name != document_name
    ]

    if remaining_chunks:
        remaining_texts = [chunk["text"] for chunk in remaining_chunks]
        embeddings = create_embeddings(remaining_texts)
        new_index = create_vector_index(embeddings)
    else:
        new_index = None

    document_chunks = remaining_chunks
    document_names = remaining_names
    vector_index = new_index

    if vector_index is not None:
        save_vector_store(vector_index, document_chunks, document_names)
    else:
        clear_vector_store()

    return {
        "message": "Document deleted successfully.",
        "documents": document_names,
        "total_documents": len(document_names),
    }


@app.delete("/clear")
def clear_documents():
    global vector_index, document_chunks, document_names

    document_chunks = []
    document_names = []
    vector_index = None

    clear_vector_store()

    return {"message": "All documents cleared successfully."}
