---
layout: default
title: BM25-Search
description: Python BM25 Tool
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>          

          
[Overview](#Overview)      
[Installation](#installation-version-011)       
[Quick Start](#quick-start)                         
[Algorithms](#algorithms)    
[Citing](#citing)         
[Release History](#release-history)         

---------------
# Overview
A collection of BM25-based algorithms, including BM25 itself, written in C++ and wrapped for Python. The following algorithms are provided in `version 0.1.1`.

- [BM25](#1-bm25-class-bm25)      
- [TF-IDF](#2-tf-idf-class-tfidf)      
- [BM11](#3-bm11-class-bm11)        
- [BM15](#4-bm15-class-bm15)             
- [BM25L](#5-bm25l-class-bm25l)       
- [BM25+](#6-bm25-class-bm25plus)           
- [BM25T](#7-bm25t-class-bm25t-beta) *(beta)*      

&nbsp;

------------

# Installation (version 0.1.1)

```
pip install bm25-search
```
- **OS**: Window, Linux (MacOs is not supported yet.)
- **Python**: Python 3.6 to 3.13

&nbsp;

------------
# Quick Start

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
Additionally, the corpus can be saved and loaded using the `save_corpus()` and *load_corpus()* methods.
```python
bm25.save_corpus("corpus.pkl", corpus)
corpus_new = bm25.load_corpus("corpus.pkl")
```

&nbsp;

------------
# Algorithms

- [BM25](#1-bm25-class-bm25)      
- [TF-IDF](#2-tf-idf-class-tfidf)      
- [BM11](#3-bm11-class-bm11)        
- [BM15](#4-bm15-class-bm15)             
- [BM25L](#5-bm25l-class-bm25l)       
- [BM25+](#6-bm25-class-bm25plus)           
- [BM25T](#7-bm25t-class-bm25t-beta) *(beta)*


## 1. BM25 `class BM25`

**BM25 (Best Matching 25)** is a ranking function used in information retrieval to score documents based on their relevance to a given query.
It is an extension of the probabilistic retrieval model and is widely used in search engines and text retrieval systems.

### Formula   

The BM25 score for a document *D* with respect to a query *Q* is given by:


 $$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}}) }$$


### Term Frequency (TF)   

Unlike raw Term Frequency, BM25 applies a saturation function, preventing a term from having an excessive influence on the score. Additionally, longer documents are adjusted so that they don’t get an unfair advantage just because they have more words.


 $$\text{TF}(D, t) = \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}}) }$$


where,       
- *f(t,D)* is the Term Frequency of term *t* in document *D*.
- *\|D\|* is the length of the document *D* (number of words).
- *avgD* is the average document length in the corpus.
- *k1* and b are hyperparameters that control term frequency saturation and document length normalization.

### Inverse Document Frequency (IDF)

The Inverse Document Frequency of term *t* typically computed as:


$$IDF(t) = \log (\frac{N - n(t) + 0.5}{n(t) + 0.5})$$


where,

- *N* is the total number of documents in the corpus.
- *n(t)* is the number of documents containing term *t*.

Through **Term frequency scaling** and **Document length normalization**, BM25 shows improved performance in retrieval tasks compared to TF-IDF, BM11, and BM15, which do not fully apply or only partially apply these strategies.

### Set BM25
```python
bm25 = BM25()
bm25.set_model(corpus, k, b)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)
- **b** (float): *b*. It adjusts the impact of document length normalization (default = 0.75)

&nbsp;

## 2. TF-IDF `class TFIDF`

**TF-IDF** is a retrieval algorithm that calculates relevance using raw Term Frequency (TF) and Inverse Document Frequency (IDF) without any normalization.
Unlike BM25, it does not adjust for document length or term saturation. 

### Formula

The TF-IDF score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{f(t, D)}{|D|}$$


### Term Frequency (TF)  

TF-IDF does not apply global length normalization or term saturation, unlike BM25.
However, it normalizes term frequency by dividing it by the document length, applying only local length normalization.


$$TF(D, t) = \frac{f(t, D)}{|D|}$$


where,       
- *f(t,D)* is the Term Frequency of term *t* in document *D*.
- *\|D\|* is the length of the document *D* (number of words).

### Inverse Document Frequency (IDF)

The Inverse Document Frequency of TF-IDF is much simpler than that of BM25.


$$IDF(D, t) = log(\frac{N}{1+n(t)})$$


where,

- *N* is the total number of documents in the corpus.
- *n(t)* is the number of documents containing term *t*.

### Set TF-IDF
```python
tfidf = TFIDF()
tfidf.set_model(corpus)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.

&nbsp;

## 3. BM11 `class BM11`
**BM11** is similar to BM25, but it differs in that it does not apply length normalization to the Term Frequency. Setting the hyperparameter *b* to 0 in BM25 makes it equivalent to BM11. Since it omits length normalization, its performance is generally lower than that of BM25. 

### Formula         

The BM11 score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1}$$


### Term Frequency (TF)  

Unlike raw Term Frequency, BM11 applies a saturation function like BM25, preventing a term from having an excessive influence on the score. However, it does not apply length normalization, unlike BM25.


$$\text{TF}(D, t) = \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1}$$


where,       
- *f(t,D)* is the Term Frequency of term *t* in document *D*.
- *\|D\|* is the length of the document *D* (number of words).
- *avgD* is the average document length in the corpus.
- *k1* is a hyperparameter that control term frequency saturation.

### Inverse Document Frequency (IDF)

The Inverse Document Frequency of BM11 is the same as that of BM25.

### Set BM11
```python
bm11 = BM11()
bm11.set_model(corpus, k)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)

&nbsp;

## 4. BM15 `class BM15`
**BM15** is similar to BM25, but it differs in that it fully applies length normalization to the Term Frequency. Setting the hyperparameter *b* to 1 in BM25 makes it equivalent to BM15. Since BM15 applies length normalization to such an extent that it ignores the difference in length, its performance is generally lower than that of BM25.

### Formula         

The BM15 score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot \frac{|D|}{\text{avgD}} }$$


### Term Frequency (TF)  

Unlike raw Term Frequency, BM15 applies full length normalization, unlike BM25, which applies it partially. However, it applies a saturation function like BM25, preventing a term from having an excessive influence on the score.


$$TF(D, t) = \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot \frac{|D|}{\text{avgD}} }$$


where,       
- *f(t,D)* is the Term Frequency of term *t* in document *D*.
- *\|D\|* is the length of the document *D* (number of words).
- *avgD* is the average document length in the corpus.
- *k1* is a hyperparameter that control term frequency saturation.

### Inverse Document Frequency (IDF)

The Inverse Document Frequency of BM15 is the same as that of BM25.

### Set BM15
```python
bm15 = BM15()
bm15.set_model(corpus, k)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)

&nbsp;

## 5. BM25L `class BM25L`
**BM25L** is an extension of BM25 that modifies Term Frequency by incorporating length normalization directly into the Term Frequency calculation.
It introduces a positive constant *δ*, preventing excessive penalization of long documents. This adjustment helps mitigate BM25’s tendency to favor shorter documents under certain conditions.

### Formula         

The BM25L score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{ (k_1 + 1) \cdot (c + \delta)}{k_1 + (c + \delta)}$$


$$c = \frac{f(t, D) }{1 - b + b \cdot \frac{|D|}{\text{avgD}}}$$


### Term Frequency (TF)  

BM25L applies length normalization directly to *f(t, D)* in both numerator and denominator.
Additionally, a positive constant *δ* is added to each normalized *f(t, D)*. 


$$\text{TF}(D, t) = \frac{ (k_1 + 1) \cdot (c + \delta)}{k_1 + (c + \delta)}$$


$$c = \frac{f(t, D) }{1 - b + b \cdot \frac{|D|}{\text{avgD}}}$$


where,       

- *f(t,D)* is the Term Frequency of term *t* in document *D*.
- *\|D\|* is the length of the document *D* (number of words).
- *avgD* is the average document length in the corpus.
- *k1* is a hyperparameter that control term frequency saturation.
- *δ* is a positive constant to prevent excessive penalization of long documents.

### Inverse Document Frequency (IDF)

The Inverse Document Frequency of BM15 is the same as that of BM25.

### Set BM25L
```python
bm25l = BM25L()
bm25l.set_model(corpus, k, b, delta)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)
- **b** (float): *b*. It adjusts the impact of document length normalization (default = 0.75)
- **delta** (float): *δ*. It alleviates the over-penalization of long documents. It must to be greater than 0(default = 1.0) 

&nbsp;

## 6. BM25+ `class BM25Plus`
**BM25+** is an extension of BM25 that introduces a positive constant *δ* as a lower bound, preventing excessive penalization of long documents. This adjustment helps mitigate BM25’s tendency to favor shorter documents under certain conditions.

### Formula

The BM25+ score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot (\frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}}) } + \delta)$$


### Term Frequency (TF)   

BM25+ introduces a positive constant *δ* as a lower bound so that Term Frequency of BM25+ is always greater than 0.  


$$TF(D, t) = \frac{ f(t, D) \cdot (k_1 + 1)}{ f(t, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}}) } + \delta$$  


### Inverse Document Frequency (IDF)

The Inverse Document Frequency of BM15 is the same as that of BM25.

### Set BM25+
```python
bm25plus = BM25Plus()
bm25plus.set_model(corpus, k, b, delta)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)
- **b** (float): *b*. It adjusts the impact of document length normalization (default = 0.75)
- **delta** (float): *δ*. It alleviates the over-penalization of long documents. It must to be greater than 0(default = 1.0) 

&nbsp;

## 7. BM25T `class BM25T` *(beta)*
**BM25T** is an extension of BM25 that introduces term-specific *k1*.
It adjusts the length-normalized Term Frequency so that its contribution reflects the proportion of documents where the term appears more frequently.

### Formula

When *Cw* is the set of all documents containing the term, the BM25T score for a document *D* with respect to a query *Q* is given by:


$$\text{score}(D, Q) = \sum_{t \in Q} IDF(t) \cdot \frac{ f(t, D) \cdot ({k_1}^{'} + 1)}{ f(t, D) + {k_1}^{'} \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}}) }$$


where,
             

$${k\_1}' = \arg \min \_{k\_1} \left(g \_{k\_1} - \frac{\sum_{t \in C\_w} \log(c) + 1}{n(t)}\right)^2$$


$$g_{k_1} = 
\begin{cases} 
\frac{k_1}{k_1 - 1} \log(k_1) & \text{if } k_1 \neq 1, \\
1 & \text{if } k_1 = 1
\end{cases}$$


$$c = \frac{f(t, D) }{1 - b + b \cdot \frac{|D|}{\text{avgD}}}$$


### Term Frequency (TF)

BM25T introduces a term-specific *k1'* for calculating Term Frequency, ensuring that the contribution of length-normalized Term Frequency is proportional to the fraction of documents where it is higher. The value of *k1'* is determined using the Newton-Raphson method. 


$$TF(D,t) = \frac{ f(t, D) \cdot ({k_1}^{'} + 1)}{ f(t, D) + {k_1}^{'} \cdot (1 - b + b \cdot \frac{|D|}{\text{avgD}})}$$


where,


$${k\_1}' = \arg \min \_{k\_1} \left(g \_{k\_1} - \frac{\sum_{t \in C\_w} \log(c) + 1}{n(t)}\right)^2$$


$$g_{k_1} = 
\begin{cases} 
\frac{k_1}{k_1 - 1} \log(k_1) & \text{if } k_1 \neq 1, \\
1 & \text{if } k_1 = 1
\end{cases}$$


$$c = \frac{f(t, D) }{1 - b + b \cdot \frac{|D|}{\text{avgD}}}$$


### Inverse Document Frequency (IDF)

The Inverse Document Frequency of BM15 is the same as that of BM25.

### Set BM25T
```python
bm25t = BM25T()
bm25t.set_model(corpus, k, b, eps, max_iter)
```

### Parameters
- **corpus** (2-dimensional list): A set of documents to be retrieved. Each document in the inner lists must be tokenized.
- **k** (float): *k1'*. It controls the influence of Term Frequency (TF) on the final score, determining how quickly the score saturates as term frequency increases.(default = 1.5)
- **b** (float): *b*. It adjusts the impact of document length normalization (default = 0.75)
- **eps** (float):  The tolerance level for convergence in Newton-Raphson optimization. An iteration breaks when the change in value is smaller than this threshold. (default = 0.05)     
- **max_iter** (int) The maximum number of iterations. If the algorithm cannot find the optimal value within max_iter iterations, the initial *k* is used as the final result. (default = 100) 

&nbsp;

----------------
# Citing

This code is based on the repository [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25), which has been referenced for the development of this tool.
```
@misc{dorianbrown2022rank_bm25,
      title={Rank-BM25: A two line search engine},
      author={Dorian Brown},
      year={2022},
      url={https://github.com/dorianbrown/rank_bm25},
}
```

If you use this **BM25-Search** library in your research or projects, please cite it as follows:
```
@misc{millet042025bm25-search,
      title={BM25-Search},
      author={Kim Minseok},
      year={2025},
      url={https://github.com/millet04/bm25_search},
}
```

&nbsp;

-----------------
# Release History
`version 0.1.0` *(2025.2.7)*          
- Provides seven BM25-based algorithms, including BM25, TF-IDF, BM11, BM15, BM25L, BM25+, and BM25T.  

`version 0.1.1` *(2025.2.15)*  
- Replace the static 'int' type variables with a dynamically allocated 'long long int' to support a larger number of documents.
- Modify some errors in the annotations and implementation details.
