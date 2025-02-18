#include "BM25T.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>

#include <unordered_map>
#include <unordered_set>
#include <math.h>
#include <numeric>
#include <algorithm>

using namespace std;
namespace py = pybind11;

/**
 * BM25T::set_tf : A function that calculates and sets the BM25T TF values for all words.
 *                 This improves the efficiency of searching for a given query.
 *
 *                 [NOTE] BM25T TF (q) = freq(q, doc) * (k'+1) / (freq(q, doc) + k' * (1 - b + b * dl / avgdl))
 *                        where k' is the term-specific optimal k found by Newton-Raphson method.  
 *
 * Parameters
 * ----------
 *    - k (double) : Saturation parameter. (default = 1.5)
 *                   Controls the influence of term frequency (TF) on the final score,
 *                   determining how quickly the score saturates as term frequency increases.
 *                   [NOTE] This k is iteratively updated during the process of finding the optimal k value.
 * 
 *    - b (double) : Length normalization parameter. (default = 0.75)
 *                   Adjusts the impact of document length normalization,
 *                   with b = 0 (BM15) ignoring length and b = 1 (BM11) fully normalizing by the average document length.
 * 
 *    - eps (double) : The tolerance level for convergence in optimization. (default = 1e-6)
 *                     The iteration breaks when the change in value is smaller than this threshold.
 * 
 *    - max_iter (int) : The maximum number of iterations. (default = 100)
 *                       If the algorithm cannot find the optimal value within max_iter iterations, the initial k is used as the final result.
 * 
 **/
void BM25T::set_tf(double k, double b, double eps, unsigned int max_iter) {

    for (const auto& word : df) {
        // Calculate 'freq(q, doc)', a frequency of a word in each document.  
        vector<double> tf_vector(*doc_n, 0.0);
        for (size_t i = 0; i < *doc_n; i++) {
            if (freq[i].find(word.first) != freq[i].end()) {
                tf_vector[i]++;
            }
        }

        // Sum c for each word (calculate the length-normalized term frequency).
        double sum_log_c = 0.0;
        for (size_t i = 0; i < *doc_n; i++) {
            double c = tf_vector[i] / (1 - b + b * (dl[i] / avgdl));
            if tf_vector[i] > 0 {
                sum_log_c += log(c);
            }
        }

        double optk = compute_optimal_k(word.first, k, sum_log_c, eps, max_iter);
        optk_set[word.first] = optk;

        // Calculates the BM25T TF values for all words. 
        for (size_t i = 0; i < *doc_n; i++) {
            tf_vector[i] = tf_vector[i] * (optk + 1) / (tf_vector[i] + optk * (1 - b + b * dl[i] / avgdl));
        }
        tf[word.first] = tf_vector;
    }
}

/**
 * BM25T::compute_optimal_k : A function that computes the optimal k for each word using Newton-Raphson method.
 *
 * Parameters
 * ----------
 *    - k (double) : Saturation parameter. (default = 1.5)
 *                   Controls the influence of term frequency (TF) on the final score,
 *                   determining how quickly the score saturates as term frequency increases.
 *                   [NOTE] This k is iteratively updated during the process of finding the optimal k value.
 *
 *    - eps (double) : The tolerance level for convergence in optimization. (default = 0.05)
 *                     The iteration breaks when the change in value is smaller than this threshold.
 *
 *    - max_iter (int) : The maximum number of iterations. (default = 100)
 *                       If the algorithm cannot find the optimal value within max_iter iterations, the initial k is used as the final result.
 *
 **/
double BM25T::compute_optimal_k(const string& word, double k, double sum_log_c, double eps, unsigned int max_iter) {

    double alpha = 1.0;
    double upper_bound = k;
    
    for (unsigned int i = 0; i < max_iter; i++) {

        // Cacluate g(x) and g'(x)
        double g_value = (k > 1) ? (k / (k - 1)) * log(k) : k;
        double g_prime = (k > 1) ? (k - 1 - log(k)) / ((k - 1) * (k - 1)) : 0;

        // Avoid zero devision.
        if (fabs(g_prime) < eps) {
            break;
        }

        // Calculate f(x) and f'(x).
        double f_value = pow((g_value - (sum_log_c + 1) / df[word]), 2);
        double f_prime = 2 * (g_value - (sum_log_c + 1) / df[word]) * g_prime;
        
        // Avoid zero devisoin.
        if (f_prime < eps) {
            break;
        }

        // Update k using Newton-Raphson method. k' = k - f(x)/f'(x)
        double k_new = k - (f_value / f_prime);

        if (fabs(k_new - k) < eps) {
            k = k_new;
            break;
        }
       
        // Update only when the difference between k' and k is smaller than the upper bound.
        // If the change in k is within an acceptable range, update k and record the new upper bound.
        if (fabs(k_new - k) < upper_bound) {
            k = k_new;
            upper_bound = fabs(k_new - k);
        }
        // If the difference is greater than the upper bound, adjust the step size using alpha.
        // This helps prevent large, unstable updates and moves towards a more stable direction.
        else {
            alpha *= 0.5;
            k_new = k - alpha * (f_value / f_prime);
            k = k_new;
        }
    }
    return k;
}

/**
 * BM25T::set_model : A function that sets the BM25T model by executing the functions 'init', 'set_idf', and 'set_tf'.
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
 *    - eps (double) : The tolerance level for convergence in optimization. (default = 0.05)
 *                     The iteration breaks when the change in value is smaller than this threshold.
 * 
 *    - max_iter (int) : The maximum number of iterations. (default = 100)
 *                       If the algorithm cannot find the optimal value within max_iter iterations, the initial k is used as the final result.
 * 
 **/
void BM25T::set_model(const vector<vector<string>>& corpus, double k, double b, double eps, unsigned int max_iter) {
    this->k = k;
    this->b = b;
    this->eps = eps;
    this->max_iter = max_iter;

    init(corpus);
    set_idf();
    set_tf(k, b, eps, max_iter);
}

/**
 * BM25T::save_model : A function that saves the BM25T model in a pickle file.
 *                     All information calculated to set the model is saved with its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to save the pickle file containing the BM25T model.
 *
 **/
void BM25T::save_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "wb");

    // Save necessary data.
    py::dict data;
    data["k"] = k;
    data["b"] = b;
    data["eps"] = eps;
    data["max_iter"] = max_iter;
    data["optk_set"] = optk_set;
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
 * BM25T::load_model : A function that loads the BM25T model from a saved pickle file.
 *                     All information required to set the model is loaded using its keys.
 *
 * Parameters
 * ----------
 *     - filepath (string)  : The path to the pickle file containing the BM25T model.
 *
 **/
void BM25T::load_model(const string& filepath) {
    // Use python pickle module.
    py::module pickle = py::module::import("pickle");
    py::module builtins = py::module::import("builtins");

    // Open file in Python
    py::object py_file = builtins.attr("open")(filepath, "rb");
    py::dict data = pickle.attr("load")(py_file);
    py_file.attr("close")();

    this->k = data["k"].cast<double>();
    this->b = data["b"].cast<double>();
    this->eps = data["eps"].cast<double>();
    this->max_iter = data["max_iter"].cast<unsigned int>();
    this->optk_set = data["optk_set"].cast<unordered_map<string, double>>();
    this->dl = data["dl"].cast<vector<size_t>>();
    this->avgdl = data["avgdl"].cast<double>();
    *this->doc_n = data["doc_n"].cast<size_t>();
    this->freq = data["freq"].cast<vector<unordered_map<string, size_t>>>();
    this->df = data["df"].cast<unordered_map<string, double>>();
    this->tf = data["tf"].cast<unordered_map<string, vector<double>>>();
    this->idf = data["idf"].cast<unordered_map<string, double>>();
}