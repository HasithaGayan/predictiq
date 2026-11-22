# PredictIQ — Day 69 Embeddings and Transformers

## Objective

Understand pretrained text embeddings,
tokenization, semantic similarity,
attention and transformer fundamentals.

## Real Pretrained Model

Model: sentence-transformers/paraphrase-MiniLM-L3-v2

Device: cpu

Embedding dimensions: 384

Query token count: 12

Measured inference duration: 0.0115 seconds

Inference timing excludes initial model download
and loading. It is not a formal speed benchmark.

## Semantic Search Experiment

Query:

Estimate hourly bicycle rental demand using weather and time.

| Document | Cosine similarity |
|---|---:|
| Bike-sharing demand varies with weather and hour of day. | 0.7746 |
| A machine-learning model predicts how many bikes are rented. | 0.5559 |
| FastAPI is a Python framework used to create web APIs. | 0.0571 |
| A chocolate cake recipe uses flour, butter and sugar. | 0.032 |

Higher cosine similarity indicates closer
vector alignment for this particular model.

Similarity is not a probability.
Similarity is not proof of factual correctness.

## How the Experiment Works

1. Read the search query and example documents.
2. Convert input text into model-readable tokens.
3. Run a pretrained transformer encoder.
4. Produce sentence-level embeddings.
5. Normalize the vectors.
6. Compute cosine similarity.
7. Sort documents by similarity.
8. Save the actual results as JSON.

## Conceptual Theory

### Tokenization

Text is split into tokens.
Each token receives a numerical ID.

Special tokens may be included.

### Embeddings

Dense numerical vectors represent text.
Semantically related text may have
similar vector directions.

This model produces 384-dimensional vectors.

### Transformer Self-Attention

Transformer attention lets tokens use
information from other tokens in context.

The core concept uses queries, keys and values:

Attention(Q,K,V) =
softmax(QK^T / sqrt(d_k))V

Q: queries
K: keys
V: values

Attention builds contextual representations.

Our experiment uses pretrained attention;
it does not train a transformer from scratch.

### Cosine Similarity

cosine(a,b) = dot(a,b) / (norm(a)*norm(b))

Values close to 1 indicate similar directions.
Values close to 0 indicate weak alignment.
Values close to -1 indicate opposite directions.

### Prompting vs Retrieval vs Fine-Tuning

Prompting:
Provide instructions or context to a model.

Retrieval:
Find relevant existing documents using similarity.

Fine-tuning:
Update model parameters using further training.

This experiment performs retrieval only.

## Verification

The automated tests cover:

- Correct cosine-similarity ranking.
- Preservation of input vectors.
- Invalid array shapes.
- Nonfinite embeddings.
- Zero-length embeddings.
- Report generation using a fake model.

The tests do not require another model download.

## Reproduction

Install project dependencies, then run:

    python -m src.predictiq.day69_embeddings

Run tests:

    python -m pytest tests -v

The initial model run needs an internet connection.
Subsequent runs can reuse cached model files.

## Local Outputs

reports/generated/day69/report.json

data/raw/huggingface_cache/

The model cache and generated results
remain outside Git tracking.

## Limitations

- Only four example documents were evaluated.
- No labeled retrieval benchmark was performed.
- Similarity does not guarantee correctness.
- Performance depends on the chosen model.
- The model was not fine-tuned.
- No live user documents were processed.
- This is an educational retrieval experiment.

## Future Application

The same embedding concept can help
KnowledgeAI retrieve relevant document sections.

## Model Reference

https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2
