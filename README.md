# BM25-Search
BM25 based algorithms, written in C++ and wrapped for Python.        

## 1. Installation
```
pip install bm25-search
```
- __OS__: Window, Linux (MacOs is not supported yet.)
- __Python__: Python 3.6 to 3.13


## 2. Usage

#### A. Set Model 
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

#### B-1. Get Scores
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

#### B-2. Get Top-K Scores & Indices


#### B-3. Get Top-K Docs

#### Save & Load




## 3. Available Algorithms



🔹`BM25`    
🔹`BM11`    
🔹`BM15`    
🔹`BM25L`    
🔹`BM25Plus`    
🔹`BM25T` (beta)    
🔹`TF-IDF`    
