#include "BM25.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>

#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <algorithm>

using namespace std;
namespace py = pybind11;

/**
 * BM25::set_tf : A function that calculates and sets the BM25 TF values for all words.
 *                This improves the efficiency of searching for a given query.
 *
 *                [NOTE] BM25 TF (q) = freq(q, doc) * (k+1) / (freq(q, doc) + k * (1 - b + b * dl / avgdl))
 *
 * Parameters
 * ----------
 *    - k (double) : Saturation parameter. (default = 2.0)
 *                   Controls the influence of term frequency (TF) on the final score,
 *                   determining how quickly the score saturates as term frequency increases.
 * 
 *    - b (double) : Length normalization parameter. (default = 0.75)
 *                   Adjusts the impact of document length normalization,
 *                   with b = 0 (BM15) ignoring length and b = 1 (BM11) fully normalizing by the average document length.
 *
 **/
void BM25::set_tf(double k, double b) {
    for (const auto& word : df) {

        // Calculate 'freq(q, doc)', a frequency of a word in each document.  
        vector<double> tf_vector(doc_n, 0.0);
        for (unsigned int i = 0; i < doc_n; i++) {
            if (freq[i].find(word.first) != freq[i].end()) {
                tf_vector[i]++;
            }
        }

        // Calculates the BM25 TF values for all words. 
        for (unsigned int i = 0; i < doc_n; i++) {
            tf_vector[i] = tf_vector[i] * (k + 1) / (tf_vector[i] + k * (1 - b + b * dl[i] / avgdl));
        }
        tf[word.first] = tf_vector;
    }
}

/**
 * BM25::set_model : A function that sets the BM25 model by executing the functions 'init', 'set_idf', and 'set_tf'.
 *
 * Parameters
 * ----------
 *     - corpus (2D vector) : A set of documents to be retrieved.
 *                            [NOTE] Each document in the inner lists must be tokenized.
 *                            [EX]   [['the',...], ['a', ...], ... , ['hello', ...]]
 * 
 *     - k (double) : Saturation parameter. (default = 2.0)
 *                    Controls the influence of term frequency (TF) on the final score,
 *                    determining how quickly the score saturates as term frequency increases.
 * 
 *     - b (double) : Length normalization parameter. (default = 0.75)
 *                    Adjusts the impact of document length normalization,
 *                    with b = 0 (BM15) ignoring length and b = 1 (BM11) fully normalizing by the average document length.
 *
 **/
void BM25::set_model(const vector<vector<string>>& corpus, double k, double b) {
    this->k = k;
    this->b = b;
    
    init(corpus);
    set_idf();
    set_tf(k, b);
}


/**
 * BM25::save_model : A function that saves the BM25 model in a pickle file.
 *                    All information calculated to set the model is saved with its keys. 
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to save the pickle file containing the BM25 model.
 *
 **/
void BM25::save_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");

    // Save necessary data.
    py::dict data;
    data["k"] = k;
    data["b"] = b;
    data["dl"] = dl;
    data["avgdl"] = avgdl;
    data["doc_n"] = doc_n;
    data["freq"] = freq;
    data["df"] = df;
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
void BM25::load_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");
    
    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    this->k = data["k"].cast<double>();
    this->b = data["b"].cast<double>();
    this->dl = data["dl"].cast<vector<int>>();
    this->avgdl = data["avgdl"].cast<double>();
    this->doc_n = data["doc_n"].cast<unsigned int>();
    this->freq = data["freq"].cast<vector<unordered_map<string, int>>>();
    this->df = data["df"].cast<unordered_map<string, double>>();
    this->tf = data["tf"].cast<unordered_map<string, vector<double>>>();
    this->idf = data["idf"].cast<unordered_map<string, double>>();
}
