#pragma once

#include "BM25T.h"
#include "CSRAdapter.h"

class BM25T_CSR : public BM25T {
public:
    CSRMatrix csr;
    void set_model(const vector<vector<string>>& corpus, double k = 1.5,
                   double b = 0.75, double eps = 0.05,
                   unsigned int max_iter = 100);
    vector<vector<double>> get_scores(const vector<vector<string>>& queries) override;
};
