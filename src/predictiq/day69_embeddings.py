
"""Day 69: real pretrained text embeddings and semantic search."""

import json
import os
import time
from pathlib import Path

import numpy as np

# Keep downloaded model files inside the ignored dataset directory.
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "raw" / "huggingface_cache"

os.environ["HF_HOME"] = str(CACHE_DIR)

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

QUERY = "Estimate hourly bicycle rental demand using weather and time."

DOCUMENTS = [
    "Bike-sharing demand varies with weather and hour of day.",
    "A machine-learning model predicts how many bikes are rented.",
    "FastAPI is a Python framework used to create web APIs.",
    "A chocolate cake recipe uses flour, butter and sugar.",
]


def rank_documents(embeddings, documents):
    """Rank documents using normalized-vector cosine similarity."""

    vectors = np.asarray(embeddings, dtype=float)

    if vectors.ndim != 2 or vectors.shape[0] != len(documents) + 1:
        raise ValueError("Invalid embedding dimensions")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain finite values")

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms <= 0):
        raise ValueError("Embeddings must have nonzero length")

    normalized = vectors / norms[:, None]

    query_vector = normalized[0]
    document_vectors = normalized[1:]

    scores = document_vectors @ query_vector

    ranked = [
        {
            "document": document,
            "similarity": round(float(score), 4),
        }
        for document, score in zip(documents, scores)
    ]

    return sorted(
        ranked,
        key=lambda item: item["similarity"],
        reverse=True,
    )


def run_day69(output_dir):
    output_dir = Path(output_dir)

    # Download once; later executions reuse the cached model.
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(CACHE_DIR / "models"),
        trust_remote_code=False,
    )

    texts = [QUERY] + DOCUMENTS

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    tokenized = model.preprocess([QUERY])
    token_count = int(
        tokenized["attention_mask"][0].sum().item()
    )

    ranked = rank_documents(embeddings, DOCUMENTS)

    report = {
        "project": "PredictIQ",
        "day": 69,
        "topic": "Pretrained embeddings and semantic similarity",
        "model": MODEL_NAME,
        "device": "cpu",
        "query": QUERY,
        "documents": len(DOCUMENTS),
        "embedding_dimensions": int(embeddings.shape[1]),
        "query_token_count": token_count,
        "inference_seconds": round(elapsed, 4),
        "ranking": ranked,
        "limitations": [
            "Similarity is not proof of factual correctness",
            "Cosine similarity is not a probability",
            "Results depend on model and example text",
            "No text-generation model is trained here",
            "No private user documents are processed",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    report = run_day69(
        ROOT / "reports" / "generated" / "day69"
    )

    print("\nPREDICTIQ DAY 69 â€” REAL TEXT EMBEDDINGS")
    print("Model:", report["model"])
    print("Embedding dimensions:", report["embedding_dimensions"])
    print("Query token count:", report["query_token_count"])
    print("Inference seconds:", report["inference_seconds"])

    print("\nSEMANTIC SIMILARITY RANKING")

    for position, item in enumerate(report["ranking"], 1):
        print(
            f'{position}. {item["similarity"]:.4f} '
            f'- {item["document"]}'
        )

    print("\nDAY 69 INITIAL EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
