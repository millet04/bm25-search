#include "BM15.h"

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
 * BM15::set_tf : A function that calculates and sets the BM15 TF values for all words.
 *                This improves the efficiency of searching for a given query.
 *
 *                [NOTE] BM15 is equivalent to BM25 when b = 0, which removes the document length normalization.
 *
 *                [NOTE] BM15 TF (q) = freq(q, doc) * (k+1) / (freq(q, doc) + k)
 *
 * Parameters
 * ----------
 *    - k (double) : Saturation parameter. (default = 1.5)
 *                   Controls the influence of term frequency (TF) on the final score,
 *                   determining how quickly the score saturates as term frequency increases.
 *
 **/
void BM15::set_tf(double k) {
    for (const auto& word : df) {

        // Calculate 'freq(q, doc)', a frequency of a word in each document.  
        vector<double> tf_vector(*doc_n, 0.0);
        for (size_t i = 0; i < *doc_n; i++) {
            if (freq[i].find(word.first) != freq[i].end()) {
                tf_vector[i]++;
            }
        }

        // Calculates the BM15 TF values for all words. 
        for (size_t i = 0; i < *doc_n; i++) {
            tf_vector[i] = tf_vector[i] * (k + 1) / (tf_vector[i] + k);
        }
        tf[word.first] = tf_vector;
    }
}

/**
 * BM15::set_model : A function that sets the BM15 model by executing the functions 'init', 'set_idf', and 'set_tf'.
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
 **/
void BM15::set_model(const vector<vector<string>>& corpus, double k) {
    this->k = k;

    init(corpus);
    set_idf();
    set_tf(k);
}

/**
 * BM15::save_model : A function that saves the BM15 model in a pickle file.
 *                    All information calculated to set the model is saved with its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to save the pickle file containing the BM15 model.
 *
 **/
void BM15::save_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");

    // Save necessary data.
    py::dict data;
    data["k"] = k;
    data["dl"] = dl;
    data["avgdl"] = avgdl;
    data["doc_n"] = *doc_n;
    data["freq"] = freq;
    data["df"] = df;
    data["tf"] = tf;
    data["idf"] = idf;

    pickle.attr("dump")(data, py_file);

    py_file.attr("close")();
}

/**
 * BM15::load_model : A function that loads the BM15 model from a saved pickle file.
 *                    All information required to set the model is loaded using its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to the pickle file containing the BM15 model.
 *
 **/
void BM15::load_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    this->k = data["k"].cast<double>();
    this->dl = data["dl"].cast<vector<size_t>>();
    this->avgdl = data["avgdl"].cast<double>();
    *this->doc_n = data["doc_n"].cast<size_t>();
    this->freq = data["freq"].cast<vector<unordered_map<string, size_t>>>();
    this->df = data["df"].cast<unordered_map<string, double>>();
    this->tf = data["tf"].cast<unordered_map<string, vector<double>>>();
    this->idf = data["idf"].cast<unordered_map<string, double>>();
}