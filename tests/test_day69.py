
"""Day 69: semantic embeddings and ranking tests."""

import json

import numpy as np
import pytest

import src.predictiq.day69_embeddings as day69


def test_cosine_ranking():
    documents = [
        "Relevant document",
        "Unrelated document",
        "Opposite document",
    ]

    vectors = np.array([
        [1.0, 0.0],   # Query
        [2.0, 0.0],   # Same direction
        [0.0, 3.0],   # Orthogonal
        [-1.0, 0.0],  # Opposite direction
    ])

    results = day69.rank_documents(vectors, documents)

    assert len(results) == 3
    assert results[0]["document"] == "Relevant document"
    assert results[0]["similarity"] == pytest.approx(1.0)

    assert results[1]["similarity"] == pytest.approx(0.0)
    assert results[2]["similarity"] == pytest.approx(-1.0)


def test_ranking_preserves_original_vectors():
    vectors = np.array([
        [3.0, 4.0],
        [6.0, 8.0],
        [4.0, -3.0],
    ])

    original = vectors.copy()

    day69.rank_documents(
        vectors,
        ["First document", "Second document"],
    )

    assert np.array_equal(vectors, original)


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 0.0]]),
        np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    ],
)
def test_invalid_embedding_shapes_rejected(vectors):
    with pytest.raises(ValueError):
        day69.rank_documents(
            vectors,
            ["Only one document"],
        )


def test_nonfinite_embeddings_rejected():
    vectors = np.array([
        [1.0, 0.0],
        [np.nan, 1.0],
    ])

    with pytest.raises(ValueError):
        day69.rank_documents(
            vectors,
            ["Document"],
        )


def test_zero_length_vectors_rejected():
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ])

    with pytest.raises(ValueError):
        day69.rank_documents(
            vectors,
            ["Document"],
        )


def test_report_generation_without_network(
    monkeypatch,
    tmp_path,
):
    """Use fake embeddings, without downloading a model."""

    class FakeModel:
        def __init__(
            self,
            model_name,
            device,
            cache_folder,
            trust_remote_code,
        ):
            assert model_name == day69.MODEL_NAME
            assert device == "cpu"
            assert trust_remote_code is False
            assert cache_folder

        def encode(self, texts, **kwargs):
            assert texts == [day69.QUERY] + day69.DOCUMENTS
            assert kwargs["normalize_embeddings"] is True

            return np.array([
                [1.0, 0.0],   # Query
                [1.0, 0.0],   # Bike-sharing weather
                [0.8, 0.6],   # Bike demand prediction
                [0.0, 1.0],   # FastAPI
                [-1.0, 0.0],  # Cake
            ])

        def preprocess(self, texts):
            assert texts == [day69.QUERY]

            return {
                "attention_mask": np.array([
                    [1, 1, 1, 0]
                ])
            }

    monkeypatch.setattr(
        day69,
        "SentenceTransformer",
        FakeModel,
    )

    output = tmp_path / "day69"

    report = day69.run_day69(output)

    assert report["embedding_dimensions"] == 2
    assert report["query_token_count"] == 3
    assert report["documents"] == 4
    assert report["device"] == "cpu"

    assert (
        report["ranking"][0]["document"]
        == day69.DOCUMENTS[0]
    )

    assert (
        report["ranking"][-1]["document"]
        == day69.DOCUMENTS[-1]
    )

    saved = output / "report.json"

    assert saved.is_file()

    loaded = json.loads(
        saved.read_text(encoding="utf-8")
    )

    assert loaded["day"] == 69
    assert loaded["ranking"] == report["ranking"]
