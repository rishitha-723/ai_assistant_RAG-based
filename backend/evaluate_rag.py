import json
from pathlib import Path

from embedding import create_query_embedding
from vector_store import create_vector_index, search_hybrid
from storage import load_vector_store


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evaluation_dataset.json"

TOP_K = 5


def load_dataset():
    """Load questions and expected relevant pages from the JSON dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATASET_PATH.name}. "
            "Place it inside the backend folder."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval():
    """Evaluate hybrid retrieval using the expected page labels."""
    dataset = load_dataset()
    vector_index, document_chunks, document_names = load_vector_store()

    if not document_chunks:
        print("No saved document chunks were found.")
        print("Upload the evaluation PDF in StudyAI first.")
        return

    target_document = dataset.get("source_document", "").strip()

    matching_chunks = [
        chunk for chunk in document_chunks
        if chunk.get("source") == target_document
    ]

    if not matching_chunks:
        print(f"Evaluation PDF not found in the saved collection: {target_document}")
        print("Documents currently saved:")
        for name in document_names:
            print(f" - {name}")
        print(
            "\nMake sure the PDF is uploaded in StudyAI and that its "
            "filename matches source_document in evaluation_dataset.json."
        )
        return

    texts = [chunk.get("text", "") for chunk in matching_chunks]

    # Build an index for only the evaluation PDF. This prevents chunks from
    # other uploaded PDFs from affecting this controlled evaluation.
    from embedding import create_embeddings
    evaluation_embeddings = create_embeddings(texts)
    evaluation_index = create_vector_index(evaluation_embeddings)

    questions = dataset.get("evaluation_questions", [])
    if not questions:
        print("The evaluation dataset contains no questions.")
        return

    precision_values = []
    recall_values = []
    reciprocal_ranks = []

    print("\nSTUDYAI — RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Evaluation document: {target_document}")
    print(f"Questions: {len(questions)}")
    print(f"Top K: {TOP_K}")
    print("=" * 60)

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        expected_pages = set(item.get("relevant_pages", []))

        query_embedding = create_query_embedding(question)

        results = search_hybrid(
            index=evaluation_index,
            query_embedding=query_embedding,
            query=question,
            text_chunks=texts,
            top_k=min(TOP_K, len(matching_chunks)),
        )

        retrieved = []
        for result in results:
            chunk_index = result["index"]
            chunk = matching_chunks[chunk_index]
            retrieved.append({
                "page": chunk.get("page"),
                "combined_score": result["combined_score"],
            })

        # Precision@K: fraction of retrieved chunks from an expected page.
        if retrieved:
            relevant_retrieved_count = sum(
                1 for result in retrieved
                if result["page"] in expected_pages
            )
            precision = relevant_retrieved_count / len(retrieved)
        else:
            precision = 0.0

        # Recall@K here is page-level: fraction of expected pages represented
        # among the retrieved chunks. It is not a count of all relevant chunks.
        retrieved_pages = {
            result["page"] for result in retrieved
            if result["page"] is not None
        }
        if expected_pages:
            recall = len(expected_pages.intersection(retrieved_pages)) / len(expected_pages)
        else:
            recall = 0.0

        # Reciprocal rank of the first retrieved chunk from an expected page.
        reciprocal_rank = 0.0
        for rank, result in enumerate(retrieved, start=1):
            if result["page"] in expected_pages:
                reciprocal_rank = 1.0 / rank
                break

        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        pages_display = sorted(retrieved_pages)
        expected_display = sorted(expected_pages)

        print(f"\n{question_id}: {question}")
        print(f"Expected relevant page(s): {expected_display}")
        print(f"Retrieved page(s):        {pages_display}")
        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"Page Recall@{TOP_K}: {recall:.2f}")
        print(f"Reciprocal rank:          {reciprocal_rank:.2f}")

    count = len(questions)
    mean_precision = sum(precision_values) / count
    mean_recall = sum(recall_values) / count
    mean_reciprocal_rank = sum(reciprocal_ranks) / count

    print("\n" + "=" * 60)
    print("OVERALL RETRIEVAL RESULTS")
    print("=" * 60)
    print(f"Mean Precision@{TOP_K}:       {mean_precision:.3f}")
    print(f"Mean Page Recall@{TOP_K}:     {mean_recall:.3f}")
    print(f"Mean Reciprocal Rank:         {mean_reciprocal_rank:.3f}")
    print("\nNote:")
    print(
        "These are page-label-based retrieval estimates. They evaluate "
        "whether retrieved chunks come from pages marked relevant in the "
        "dataset; they do not measure Gemini answer correctness."
    )


if __name__ == "__main__":
    evaluate_retrieval()
