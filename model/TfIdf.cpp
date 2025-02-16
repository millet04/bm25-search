#include "TfIdf.h"

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
 * TFIDF::set_idf : A function that calculates and sets the TF-IDF's IDF values for all words.
 *                  This improves the efficiency of searching for a given query.
 *
 *                  [NOTE] TF-IDF IDF (q) = log(doc_n / (1 + df(q)))
 **/
void TFIDF::set_idf(void) {
    // Efficiently calculate IDF using logarithmic operations.
    for (const auto& word : df) {
        double idf_score = log(doc_n) - log(1 + word.second);
        idf[word.first] = idf_score;
    }
}

/**
 * TFIDF::set_tf : A function that calculates and sets the TF-IDF's TF values for all words.
 *                 This improves the efficiency of searching for a given query.
 *
 *                [NOTE] TF-IDF TF (q) = freq(q, doc) / dl
 *
 **/
void TFIDF::set_tf(void) {
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
            tf_vector[i] = tf_vector[i] / dl[i];
        }
        tf[word.first] = tf_vector;
    }
}

/**
 * TFIDF::set_model : A function that sets the TFIDF model by executing the functions 'init', 'set_idf', and 'set_tf'.
 *
 * Parameters
 * ----------
 *     - corpus (2D vector) : A set of documents to be retrieved.
 *                            [NOTE] Each document in the inner lists must be tokenized.
 *                            [EX]   [['the',...], ['a', ...], ... , ['hello', ...]]
 *
 **/
void TFIDF::set_model(const vector<vector<string>>& corpus) {
    init(corpus);
    set_idf();
    set_tf();
}


/**
 * TFIDF::save_model : A function that saves the TFIDF model in a pickle file.
 *                     All information calculated to set the model is saved with its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to save the pickle file containing the TFIDF model.
 *
 **/
void TFIDF::save_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");

    // Save necessary data.
    py::dict data;
    data["dl"] = dl;
    data["doc_n"] = doc_n;
    data["freq"] = freq;
    data["df"] = df;
    data["tf"] = tf;
    data["idf"] = idf;

    pickle.attr("dump")(data, py_file);

    py_file.attr("close")();
}

/**
 * TFIDF::load_model : A function that loads the TFIDF model from a saved pickle file.
 *                     All information required to set the model is loaded using its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to the pickle file containing the TFIDF model.
 *
 **/
void TFIDF::load_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    this->dl = data["dl"].cast<vector<int>>();
    this->doc_n = data["doc_n"].cast<unsigned int>();
    this->freq = data["freq"].cast<vector<unordered_map<string, int>>>();
    this->df = data["df"].cast<unordered_map<string, double>>();
    this->tf = data["tf"].cast<unordered_map<string, vector<double>>>();
    this->idf = data["idf"].cast<unordered_map<string, double>>();
}