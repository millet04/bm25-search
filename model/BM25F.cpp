#include "BM25F.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>

#include <unordered_set>
#include <numeric>
#include <algorithm>

using namespace std;
namespace py = pybind11;


/**
 * BM25F::set_tf : A function that calculates and sets the BM25F TF values for all words.
 *                 This improves the efficiency of searching for a given query.
 *
 *                [NOTE] BM25F TF (q) = (k + 1) * FREQ(q, doc) / (k +  FREQ(q, doc))
 *                      
 *                             FREQ(q, doc) = (w_1 * freq_1(q, field_1) / B_1) + (w_2 * freq_2(q, field_2) / B_2) + ... 
 *                      
 *                             B_n = (1 - b_n) + b_n * (dl_n / avgdl_n)
 * 
 * Parameters
 * ----------
 *    - field_n (int) : Number of fields. 
 * 
 *    - k (double) : Saturation parameter. (default = 1.5)
 *                   Controls the influence of term frequency (TF) on the final score,
 *                   determining how quickly the score saturates as term frequency increases.
 *
 *    - b (1D vector) : Length normalization parameter of each field. 
 *                      Adjusts the impact of document length normalization,
 *                      with b = 0 (BM15) ignoring length and b = 1 (BM11) fully normalizing by the average document length.
 * 
 *    - w (1D vector) : Weights of each field. 
 *                      The higher the field weight, the more it affects the total BM25F score.
 **/

void BM25F::set_tf(size_t field_n, double k, vector<double> b, vector<double> w) {

    // Iterate over all fields.
    for (size_t i = 0; i < field_n; i++) {

        for (const auto& word : field_df[i]) {

            // Calculate 'freq(q, field)', the frequency of a word in each document of the current field.
            vector<double> tf_vector(*doc_n, 0.0);
            for (size_t j = 0; j < *doc_n; j++) {
                if (field_freq[i][j].find(word.first) != field_freq[i][j].end()) {
                    tf_vector[j]++;
                }
            }

            // Apply normalization for each document in the current field.
            for (size_t j = 0; j < *doc_n; j++) {
                tf_vector[j] = (w[i] * tf_vector[j]) / (1 - b[i] + b[i] * field_dl[i][j] / field_avgdl[i]);
            }

            // If the word is already in 'tf' (appears in the second or later fields), update its existing entry.
            if (tf.find(word.first) != tf.end()) {
                for (size_t j = 0; j < *doc_n; j++) {
                    tf[word.first][j] += tf_vector[j];
                }
            }
            // If the word is not in 'tf' (first occurrence in the first field), initialize its 'tf' entry.
            else {
                tf[word.first] = tf_vector;
            }
        }
    }
    // Calculate the final TF score of each word.
    for (const auto& word : tf) {
        for (size_t j = 0; j < *doc_n; j++) {
            tf[word.first][j] = ((k + 1) * tf[word.first][j]) / (k + tf[word.first][j]);
        }
    }
}

/**
 * BM25F::set_idf : A function that calculates and sets the BM25F IDF values for all words.
 *                 This improves the efficiency of searching for a given query.
 *
 *                 [NOTE] BM25F IDF (q) = log(((doc_n - df(q) + 0.5) / (df(q) + 0.5)) + 1)
 **/
void BM25F::set_idf(void) {    
    // Efficiently calculate IDF using logarithmic operations.
    for (const auto& word : total_df) {
        double idf_score = log(*doc_n + 1) - log(word.second + 0.5);
       idf[word.first] = idf_score;
    }
}

/**
 * BM25F::set_model : A function that sets the BM25F model by executing the functions 'init', 'set_idf', and 'set_tf'.
 *
 * Parameters
 * ----------
 *     - corpus (2D vector) : A set of documents to be retrieved.
 *                            [NOTE] Each document in the inner lists must be tokenized.
 *                            [EX]   [['the',...], ['a', ...], ... , ['hello', ...]]
 *
 *     - k (double) : Saturation parameter. (default = 1.5)
 *                    Controls the influence of term frequency (TF) on the final score,
 *                    determining how quickly the score saturates as term frequency increases.
 *
 *     - b (double) : Length normalization parameter. (default = 0.75)
 *                    Adjusts the impact of document length normalization,
 *                    with b = 0 (BM15) ignoring length and b = 1 (BM11) fully normalizing by the average document length.
 *
 **/
void BM25F::set_model(const vector<vector<vector<string>>>&corpus, double k, vector<double> b, vector<double> w) {

    size_t field_n = corpus.size();

    // Adjust the number of b to match the number of fields.
    while (b.size() > field_n) {
        b.pop_back();
    }

    while (b.size() < field_n) {
        b.push_back(0.75);
    }

    // Adjust the number of w to match the number of fields.
    while (w.size() > field_n) {
        w.pop_back();
    }

    while (w.size() < field_n) {
        w.push_back(1.0);
    }

    this->k = k;
    this->b = b;
    this->w = w;

    // Initialize BM25F statistics for each field.
    for (const auto& field : corpus) {
        init(field);
        field_dl.push_back(this->dl);
        field_avgdl.push_back(this->avgdl);
        field_df.push_back(this->df);
        field_freq.push_back(this->freq);

        // Set the total 'df' to calculate the IDF score based on it.
        for (const auto& word : df) {
            total_df[word.first]++;
        }

        this->dl.clear();
        this->avgdl = 0.0;
        this->df.clear();
        this->freq.clear();
    }

    set_tf(field_n, k, b, w);
    set_idf();
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
 *     - docs (1D vector) : A set of documents, which is composed of multiple fields, to be retrieved.
 *                          A type of the documents must be 'unordered map<string, string>'
 *                          
 *                         [NOTE] Each field of document here must not be tokenized since this method returns entire sentences.
 *
 *     - n (int) : The number of documents to be retrieved.
 *
 * Returns
 * -----------
 *     - output (2D vector) : A set of top-k documents retrieved.
 *                            [[{'title':"", 'text':""}, {'title':"", 'text':""}, ...], [{'title':"", 'text':""}, {'title':"", 'text':""}, ...] ]
 *
 **/

vector<vector<unordered_map<string, string>>> BM25F::get_topk_docs(const vector<vector<string>>& queries, const vector<unordered_map<string, string>>& docs, size_t n) {
    vector<vector<double>> scores = get_scores(queries);

    if (n > *doc_n) {
        throw out_of_range("Parameter 'n' exceeds the document length.");
    }

    vector<vector<unordered_map<string, string>>> output;

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
        vector<unordered_map<string, string>> out_docs;
        for (size_t i : indices) {
            out_docs.push_back(docs[i]);
        }
        output.push_back(out_docs);
    }
    return output;
}



/**
 * BM25::save_model : A function that saves the BM25F model in a pickle file.
 *                    All information calculated to set the model is saved with its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to save the pickle file containing the BM25 model.
 *
 **/
void BM25F::save_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");

    // Save necessary data.
    py::dict data;
    data["k"] = k;
    data["b"] = b;
    data["w"] = w;
    data["doc_n"] = *doc_n;
    data["field_dl"] = field_dl;
    data["field_avgdl"] = field_avgdl;
    data["field_freq"] = field_freq;
    data["field_df"] = field_df;
    data["total_df"] = total_df;

    data["tf"] = tf;
    data["idf"] = idf;

    pickle.attr("dump")(data, py_file);

    py_file.attr("close")();
}

/**
 * BM25::load_model : A function that loads the BM25 model from a saved pickle file.
 *                    All information required to set the model is loaded using its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to the pickle file containing the BM25 model.
 *
 **/
void BM25F::load_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    this->k = data["k"].cast<double>();
    this->b = data["b"].cast<vector<double>>();
    this->w = data["w"].cast<vector<double>>();
    this->field_dl = data["field_dl"].cast<vector<vector<size_t>>>();
    this->field_avgdl = data["field_avgdl"].cast<vector<double>>();
    *this->doc_n = data["doc_n"].cast<size_t>();
    this->field_freq = data["field_freq"].cast<vector<vector<unordered_map<string, size_t>>>>();
    this->total_df = data["total_df"].cast<unordered_map<string, double>>();
    this->field_df = data["field_df"].cast<vector<unordered_map<string, double>>>();
    this->tf = data["tf"].cast<unordered_map<string, vector<double>>>();
    this->idf = data["idf"].cast<unordered_map<string, double>>();
}

