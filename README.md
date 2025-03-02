# BM25-Search
A collection of BM25 based algorithms, including BM25 itself, written in C++ and wrapped for Python.  
The following algorithms are provided in `version 0.2.0`.

- BM25
- TF-IDF   
- BM11   
- BM15    
- BM25L    
- BM25+           
- BM25T
- BM25F   

&nbsp;

For detailed information on the algorithm and usage, refer to the following document:
[https://millet04.github.io/bm25-search/](https://millet04.github.io/bm25-search/)



## 1. Installation
```
pip install bm25-search
```
- __OS__: Window, Linux (MacOs is not supported yet.)
- __Python__: Python 3.6 to 3.13

 
## 2. Quick Start

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

## 3. Other Algorhithms
The following algorithms are provided, with the same usage, but different parameters for `set_model()` method. You can use them by creating an instance like `bm25plus = BM25Plus()` and following the guidance below regarding the parameters of the `set_model()` method.

- BM25 `BM25` ➡️ ```bm25.set_model(corpus, k=1.5, b=0.75)```  
- TF-IDF `TFIDF` ➡️ ```tf_idf.set_model(corpus)```  
- BM11 `BM11` ➡️ ```bm11.set_model(corpus, k=1.5)```         
- BM15 `BM15` ➡️ ```bm15.set_model(corpus, k=1.5)```         
- BM25L `BM25L` ➡️ ```bm25l.set_model(corpus, k=1.5, b=0.75, delta = 1.0)```           
- BM25+ `BM25Plus` ➡️ ```bm25plus.set_model(corpus, k=1.5, b=0.75, delta = 1.0)```       
- BM25T `BM25T` ➡️ ```bm25t.set_model(corpus, k=1.5, b=0.75, eps=0.05, max_iter=100)```


## 4. BM25F
The BM25F algorithm computes scores for each field and combines them using weights to produce the final score.
Since the scoring is field-based, a document must have multiple fields.

```python
# Each document must have multiple fields, such as 'title' and 'text'.
corpus = [
    {'title': "Morning Routine",
     'text': "I wake up early and drink a cup of coffee"},
    {'title': "A Rainy Day",
     'text': "She gets lost in the pages of her favorite novel"},
    {'title':" Lost in a Book",
     'text': "She gets lost in the pages of her favorite novel"},
    {'title':"A Walk in the Park",
     'text':"Birds chirp as I stroll through the quiet park"},
    {'title':"Weekend Plans",
     'text':"We will go to the beach this Saturday"}
]

# The text in each document's fields should be grouped by field.
corpus_tokenized = [[] for _ in range(2)]

for doc in corpus:
    corpus_tokenized[0].append(doc['title'].lower().split())
    corpus_tokenized[1].append(doc['text'].lower().split())

bm25f = BM25F()

# TF and IDF calculations are done at this stage, 
# so this might take a while if the corpus is large.
bm25f.set_model(corpus_tokenized, k=1.5, b=[0.75, 0.75], w=[3.0, 1.0])
```
The parameters(*corpus*) of method `get_topk_docs()` of `BM25F` should be of dictionary type. Therefore, it's better to process documents as dictionaries, as shown in the example above.   

```
bm25f.get_topk_docs(queries_tokenized, corpus, n=2)
```

&nbsp;

## Citing
This code is referenced from the repository [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25).
```
@misc{dorianbrown2022rank_bm25,
      title={Rank-BM25: A two line search engine},
      author={Dorian Brown},
      year={2022},
      url={https://github.com/dorianbrown/rank_bm25},
}
```

If you use this **BM25-Search** in your research or projects, please cite it as follows:
```
@misc{millet042025bm25-search,
      title={BM25-Search},
      author={Kim Minseok},
      year={2025},
      url={https://github.com/millet04/bm25_search},
}
```