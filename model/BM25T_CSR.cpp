#include "BM25T_CSR.h"

void BM25T_CSR::set_model(const vector<vector<string>>& corpus, double k,
                          double b, double eps, unsigned int max_iter) {
    BM25T::set_model(corpus, k, b, eps, max_iter);
    csr.build(tf, idf, *doc_n);
}

vector<vector<double>> BM25T_CSR::get_scores(const vector<vector<string>>& queries) {
    return csr.scores(queries);
}
