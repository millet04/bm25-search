<h1 align="center">BM25-Search</h1>

[한국어 README](README.ko.md)

BM25-Search is a Python library that provides BM25-family ranking algorithms implemented in C++ and exposed through pybind11. It supports dense TF matrices for compatibility and CSR-backed variants for memory-efficient search.

## Features

- BM25-family algorithms (BM25, BM11, BM15, BM25L, BM25+, BM25T, and BM25F) and TF-IDF retrieval
- CSR variants for every algorithm (`*_CSR`)
- Precomputed term frequency and inverse document frequency values
- Score search, top-k ranking, and top-k document retrieval
- Model and corpus persistence through pickle

## Installation

```bash
pip install bm25-search
```

- Supported operating systems: Windows and Linux
- Supported Python versions: 3.6–3.13
- macOS is not currently supported

## Quick start

Documents and queries must be tokenized as lists of tokens.

```python
from bm25_search import BM25

corpus = [
    "The sun is shining brightly",
    "It is raining now",
    "Snow is expected tonight",
]
corpus_tokens = [document.lower().split() for document in corpus]

model = BM25()
model.set_model(corpus_tokens, k=1.5, b=0.75)

queries = [["snow"], ["sun", "shining"]]
scores = model.get_scores(queries)
top_scores, top_indices = model.get_topk(queries, n=2)
top_documents = model.get_topk_docs(queries, corpus, n=2)
```

`set_model()` performs preprocessing. Reuse the same model for multiple queries to avoid recalculating TF and IDF values.

## CSR search

CSR classes expose the same search methods as their dense counterparts while storing the calculated TF matrix in compressed sparse row form.

```python
from bm25_search import BM25_CSR

model = BM25_CSR()
model.set_model(corpus_tokens, k=1.5, b=0.75)
scores = model.get_scores([["snow"]])
```

Available CSR classes are `BM25_CSR`, `TFIDF_CSR`, `BM11_CSR`, `BM15_CSR`, `BM25L_CSR`, `BM25Plus_CSR`, `BM25T_CSR`, and `BM25F_CSR`.

## Algorithms

All dense classes use `set_model()` followed by `get_scores()`, `get_topk()`, or `get_topk_docs()`.

| Algorithm | Class | Main parameters |
| --- | --- | --- |
| BM25 | `BM25` | `k=1.5`, `b=0.75` |
| TF-IDF | `TFIDF` | — |
| BM11 | `BM11` | `k=1.5` |
| BM15 | `BM15` | `k=1.5` |
| BM25L | `BM25L` | `k=1.5`, `b=0.75`, `delta=1.0` |
| BM25+ | `BM25Plus` | `k=1.5`, `b=0.75`, `delta=1.0` |
| BM25T | `BM25T` | `k=1.5`, `b=0.75`, `eps=0.05`, `max_iter=100` |
| BM25F | `BM25F` | `k=1.5`, field `b`, field `w` |

For algorithm definitions and formulas, see the [full documentation](docs/index.md).

### BM25F input

BM25F expects a three-dimensional, field-major tokenized corpus. The outer list contains fields; each field contains one tokenized list per document.

```python
from bm25_search import BM25F

corpus = [
    [["morning", "routine"], ["rainy", "day"]],  # title field
    [["coffee", "early"], ["favorite", "novel"]],  # body field
]

model = BM25F()
model.set_model(corpus, k=1.5, b=[0.75, 0.75], w=[3.0, 1.0])
```

## Save and load

```python
model.save_model("bm25.pkl")

restored = BM25()
restored.load_model("bm25.pkl")
```

Corpora can also be saved and loaded with `save_corpus()` and `load_corpus()`.

## License and attribution

This project is released under the license in [LICENSE](LICENSE). Some implementation ideas are based on [rank_bm25](https://github.com/dorianbrown/rank_bm25) and the paper *Improvements to BM25 and Language Models Examined*.

```bibtex
@misc{millet042025bm25-search,
  title={BM25-Search},
  author={Kim Minseok},
  year={2025},
  url={https://github.com/millet04/bm25_search}
}
```
