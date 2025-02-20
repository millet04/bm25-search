/**
 * This code is based on code from the 'dorianbrown/rank_bm25' repository, which is licensed under the Apache License 2.0.
 *
 * Original repository: https://github.com/dorianbrown/rank_bm25
 * License: Apache License 2.0
 *
 * The full text of the license can be found at:
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 **/

#include "Base.h"

#include <numeric>
#include <algorithm>
 
using namespace std;
namespace py = pybind11;
 
 
/**
 * Base::init : A function that initializes all the attributes including 'doc_n', 'avgdl',
 *              'dl', df', and 'tf' when a class inheriting this 'Base' class is declared.
 *
 * Parameters
 * ----------
 *     - corpus (2D vector) : A set of documents to be retrieved.
 *                            [NOTE] Each document in the inner lists must be tokenized.
 *                            [EX]   [['the',...], ['a', ...], ... , ['hello', ...]]
 **/
void Base::init(const vector<vector<string>>&corpus) {
    // The total number of documents.
    *doc_n = corpus.size();
 
    // Total length of documents. It is used to caluculate 'avgdl'.
    size_t suml = 0;
 
    for (const auto& doc : corpus) {
        // Add length of each doucment to get 'suml'. 
        // Push length of each document into 'dl'.
        suml += doc.size();
        dl.push_back(doc.size());
 
        // Push the number of times a token appears in a specific document into 'freq'. 
        unordered_map<string, size_t> doc_freq;
        for (const auto& token : doc) {
            doc_freq[token]++;
        }
        freq.push_back(doc_freq);
 
        // Push the number of documents that contain a specific word into 'df'.
        unordered_set<string> words(doc.begin(), doc.end());
        for (const auto& word : words) {
            df[word]++;
        }
    }
    // The average length of all documents.
    avgdl = static_cast<double>(suml) / corpus.size();
}
 
 
/**
 * Base::set_idf : A function that calculates and sets the BM25 IDF values for all words.
 *                 This improves the efficiency of searching for a given query.
 *
 *                 [NOTE] BM25 IDF (q) = log(((doc_n - df(q) + 0.5) / (df(q) + 0.5)) + 1)
 **/
void Base::set_idf(void) {
    // Efficiently calculate IDF using logarithmic operations.
    for (const auto& word : df) {
        double idf_score = log(*doc_n + 1) - log(word.second + 0.5);
       idf[word.first] = idf_score;
    }
}
 
 
/**
 * Base::get_scores : A function that returns BM25 scores for all words.
 *                    It updates the 'scores' attribute.
 *
 * Parameters
 * ----------
 *     - queries (2D vector) : A set of queries to be processed.
 *                             [NOTE] Each query in the inner lists must be tokenized.
 *                             [EX]   [['what',...], ['where', ...], ... , ['who', ...]]
 * Returns
 * ----------
 *     - scores (2D vector) : A set of relevance scores corresponding to each query.
 *
 **/
vector<vector<double>> Base::get_scores(const vector<vector<string>>& queries) {
    vector<vector<double>> scores;
    for (const auto& query : queries) {
 
        // The cumulative score of all tokens becomes the final score of the query.
        vector<double> score(*doc_n, 0.0);
        for (const auto& token : query) {
            if (tf.find(token) != tf.end()) {
 
                // Calculate final score based on tf and idf scores.
                for (size_t i = 0; i < *doc_n; i++) {
                    score[i] += tf[token][i] * idf[token];
                }
            }
            else {
                continue;
            }
        }
        scores.push_back(score);
    }
    return scores;
}
 
 
/**
 * Base::get_topk : A function that returns scores and indices for top-k documents.
 *
 * Parameters
 * ----------
 *     - queries (2D vector) : A set of queries to be processed.
 *                             [NOTE] Each query in the inner lists must be tokenized.
 *                             [EX]   [['what',...], ['where', ...], ... , ['who', ...]]
 * 
 *     - n (int) : The number of documents to be retrieved.
 *
 * Returns
 * ----------
 *     - output (pair) : A pair of scores vector and indices vector for the top-k documents, sorted by descending scores.
 *                       [EX] [[3.xxx, 2.xxx, 1.xxx], [25675, 1845, 76542]]
 *
 **/
pair<vector<vector<double>>, vector<vector<size_t>>> Base::get_topk(const vector<vector<string>>& queries, size_t n) {
    vector<vector<double>> scores = get_scores(queries);
 
    if (n > *doc_n) {
        throw out_of_range("Parameter 'n' exceeds the document length.");
    }
 
    vector<vector<double>> out_scores;
    vector<vector<size_t>> out_indices;
 
    for (const auto& score : scores) {
         
        vector<double> temp_scores;
        vector<size_t> temp_indices;
         
        vector<size_t> indices(score.size());
        iota(indices.begin(), indices.end(), 0);
 
        // Sort by descending scores.
        sort(indices.begin(), indices.end(), [&score](size_t i1, size_t i2) {
            return score[i1] > score[i2];
            });
 
        // Get top-k indices and keep only top-k.
        if (n < indices.size()) {
            indices.resize(n);
        }
 
        // Prepare output in the format.
        for (size_t i : indices) {
            temp_scores.push_back(static_cast<double>(score[i]));
            temp_indices.push_back(static_cast<int>(i));
        }
 
        out_scores.push_back(temp_scores);
        out_indices.push_back(temp_indices);
    }
    pair<vector<vector<double>>, vector<vector<size_t>>> output = { out_scores, out_indices };
    return output;
}
 
/**
 * Base::get_topk_docs : A function that returns top-k documents.
 *
 * Parameters
 * ----------
 *     - queries (2D vector) : A set of queries to be processed.
 *                             [NOTE] Each query in the inner lists must be tokenized.
 *                             [EX]   [['what',...], ['where', ...], ... , ['who', ...]]
 * 
 *     - docs (1D vector) : A set of documents to be retrieved.
 *                          [NOTE] Each document here must not be tokenized since this method returns entire sentences.
 * 
 *     - n (int) : The number of documents to be retrieved.
 *
 * Returns
 * -----------
 *     - output (2D vector) : A set of top-k documents retrieved.
 *                            [['This is ...', 'The men ...'], ['When I ...', 'In last ...'], ... ['One of ....', 'The most ...']]
 *
 **/
vector<vector<string>> Base::get_topk_docs(const vector<vector<string>>& queries, const vector<string>& docs, size_t n) {
    vector<vector<double>> scores = get_scores(queries);
 
    if (n > *doc_n) {
        throw out_of_range("Parameter 'n' exceeds the document length.");
    }
 
    vector<vector<string>> output;
 
    for (const auto& score : scores) {
        vector<size_t> indices(score.size());
        iota(indices.begin(), indices.end(), 0);
 
        // Sort by descending scores.
        sort(indices.begin(), indices.end(), [&score](size_t i1, size_t i2) {
            return score[i1] > score[i2];
            });
 
        // Get top-n indices and keep only top-n.
        if (n < indices.size()) {
            indices.resize(n);
        }
 
        // Prepare output in the format.
        vector<string> out_docs;
        for (size_t i : indices) {
            out_docs.push_back(docs[i]);
        }
        output.push_back(out_docs);
    }
    return output;
}
 
/**
 * Base::save_corpus : A function that saves a corpus in a pickle file.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : A path to save the corpus.
 * 
 *     - corpus (1D vector) : A corpus to be saved.
 *                            [Note] This function saves the corpus that is not tokenized.
 *
 **/
void Base::save_corpus(const string& filepath, const vector<string>& corpus) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");
     
    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");
 
    // Save corpus that is not tokenized.
    py::dict data;
    data["corpus"] = corpus;
 
    pickle.attr("dump")(data, py_file);
 
    py_file.attr("close")();
}
 
py::list Base::load_corpus(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");
 
    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    return data["corpus"];
}
 
void Base::set_tf(double k, double b) {
    // Empty function for overriding.
}
 
void Base::set_model(const vector<vector<string>>& corpus, double k, double b) {
    // Empty function for overriding.
}
 
void Base::save_model(const string& filepath) {
    // Empty function for overriding.
}
 
void Base::load_model(const string& filepath) {
    // Empty function for overriding.
}