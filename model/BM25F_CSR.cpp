#include "BM25F_CSR.h"

void BM25F_CSR::set_model(const vector<vector<vector<string>>>& corpus, double k,
                          vector<double> b, vector<double> w) {
    BM25F::set_model(corpus, k, b, w);
    csr.build(tf, idf, *doc_n);
}

vector<vector<double>> BM25F_CSR::get_scores(const vector<vector<string>>& queries) {
    return csr.scores(queries);
}
