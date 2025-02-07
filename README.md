# BM25-Search
BM25 based algorithms, written in C++ and wrapped for Python.        

## 1. Installation
```
pip install bm25-search
```
- __OS__: Window, Linux (MacOs is not supported yet.)
- __Python__: Python 3.6 to 3.13


## 2. Usage

### 1️⃣ Set Model 
Calculate the TF (Term Frequency) and IDF (Inverse Document Frequency) for all tokens in the given documents using the `set_model()` method. This pre-calculation helps reduce latency during the search stage. 

```python
from bm25_search import BM25

corpus = [
    "The sun is shining brightly",
    "It is raining now",
    "The breeze feels cool",
    "Snow is expected tonight",
    "The sky is cloudy"    
]

# Tokenized 2-dimensional list 
corpus_tokenized = [doc.lower().split(" ") for doc in corpus]

bm25 = BM25()

# TF and IDF calculations are done at this stage, 
# so this might take a while if the corpus is large. 
bm25.set_model(corpus_tokenized, k=1.5, b=0.75)
```

### 2️⃣ Get Scores
You can obtain the scores of all documents for the given queries using the `get_scores()` method.

```python
# Tokenized 2-dimensional list
queries = ["white snow", "cloudy sky"]
queries_tokenized = [query.lower().split(" ") for query in queries]

bm25.get_scores(queries_tokenized)
```
```
[[0.0, 0.0, 0.0, 1.4166511719473336, 0.0],
 [0.0, 0.0, 0.0, 0.0, 2.833302343894667]]
```

### 3️⃣ Get Top-K Scores & Indices
You can obtain the scores and indices of top-k documents for the given queries using the `get_topk()` method.
```python
bm25.get_topk(queries_tokenized, n=2)
```

```
([[1.4166511719473336, 0.0], [2.833302343894667, 0.0]], [[3, 0], [4, 0]])
```

### 4️⃣ Get Top-K Docs
You can obtain the top-k documents for the given queries using the `get_topk_docs()` method.
```python
bm25.get_topk_docs(queries_tokenized, corpus, n=2)
```
```
[['Snow is expected tonight', 'The sun is shining brightly'],
 ['The sky is cloudy', 'The sun is shining brightly']]
```

### 5️⃣ Save & Load
You can save the model to a pickle file using the `save_model()` method. All statistics required to build the model are saved. 
```python
bm25.save_model("mybm25.pkl")
```
You can load the saved model using the `load_model()` method.
```python
bm25_new = BM25()
bm25_new.load_model("mybm25.pkl")
```
Additionally, the corpus can be saved and loaded using the `save_corpus()` and `load_corpus()` methods.
```python
bm25.save_corpus("corpus.pkl", corpus)
corpus_new = bm25.load_corpus("corpus.pkl")
```

## 3. Available Algorithms



🔹`BM25`    
🔹`BM11`    
🔹`BM15`    
🔹`BM25L`    
🔹`BM25Plus`    
🔹`BM25T` (beta)    
🔹`TF-IDF`    
