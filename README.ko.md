<h1 align="center">BM25-Search</h1>

[English README](README.md)

BM25-Search는 C++로 구현되었으며, pybind11을 통해 Python에서 사용할 수 있는 BM25 계열 검색 라이브러리입니다. 기존 호환성을 위한 dense TF 행렬 구현과 메모리 효율적인 검색을 위한 CSR 구현을 모두 제공합니다.

## Features

- BM25 계열 알고리즘(BM25, BM11, BM15, BM25L, BM25+, BM25T, BM25F)과 TF-IDF 검색
- 모든 알고리즘의 CSR 버전(`*_CSR`)
- term frequency와 inverse document frequency의 사전 계산
- 점수 계산, 상위 k개 점수와 문서 인덱스 조회, 상위 k개 문서 조회
- pickle을 이용한 모델 및 코퍼스 저장과 복원

## Installation

```bash
pip install bm25-search
```

- 지원 운영체제: Windows, Linux
- 지원 Python 버전: 3.6–3.13
- macOS는 현재 지원하지 않습니다.

## Quick start

문서와 질의는 토큰화된 리스트로 제공해야 합니다.

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

`set_model()`을 호출하면 필요한 전처리가 수행됩니다. 여러 질의를 처리할 때 같은 모델을 재사용하면 TF와 IDF를 반복해서 계산하지 않아도 됩니다.

## CSR search

CSR 클래스는 dense 버전과 동일한 검색 메서드를 제공하며, 계산된 TF 행렬을 compressed sparse row(CSR) 형식으로 저장합니다.

```python
from bm25_search import BM25_CSR

model = BM25_CSR()
model.set_model(corpus_tokens, k=1.5, b=0.75)
scores = model.get_scores([["snow"]])
```

사용할 수 있는 CSR 클래스는 `BM25_CSR`, `TFIDF_CSR`, `BM11_CSR`, `BM15_CSR`, `BM25L_CSR`, `BM25Plus_CSR`, `BM25T_CSR`, `BM25F_CSR`입니다.

## Algorithms

모든 dense 버전은 `set_model()` 호출 후 `get_scores()`, `get_topk()`, `get_topk_docs()` 메서드를 사용합니다.

| 알고리즘 | 클래스 | 주요 파라미터 |
| --- | --- | --- |
| BM25 | `BM25` | `k=1.5`, `b=0.75` |
| TF-IDF | `TFIDF` | — |
| BM11 | `BM11` | `k=1.5` |
| BM15 | `BM15` | `k=1.5` |
| BM25L | `BM25L` | `k=1.5`, `b=0.75`, `delta=1.0` |
| BM25+ | `BM25Plus` | `k=1.5`, `b=0.75`, `delta=1.0` |
| BM25T | `BM25T` | `k=1.5`, `b=0.75`, `eps=0.05`, `max_iter=100` |
| BM25F | `BM25F` | 필드별 `b`, 필드별 `w` |

알고리즘 정의와 수식은 [전체 문서](docs/index.md)를 참고하세요.

### BM25F input

BM25F는 필드 기준으로 구성된 3차원 토큰 코퍼스를 받습니다. 가장 바깥쪽 리스트가 필드를 나타내며, 각 필드에는 문서별 토큰 리스트가 들어갑니다.

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

코퍼스도 `save_corpus()`와 `load_corpus()`를 사용해 저장하고 복원할 수 있습니다.

## License and attribution

프로젝트 라이선스는 [LICENSE](LICENSE)에서 확인할 수 있습니다. 일부 구현은 [rank_bm25](https://github.com/dorianbrown/rank_bm25) 저장소와 *Improvements to BM25 and Language Models Examined* 논문을 참고했습니다.

```bibtex
@misc{millet042025bm25-search,
  title={BM25-Search},
  author={Kim Minseok},
  year={2025},
  url={https://github.com/millet04/bm25_search}
}
```
