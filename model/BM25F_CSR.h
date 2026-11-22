#pragma once

#include "BM25F.h"
#include "CSRAdapter.h"

class BM25F_CSR : public BM25F {
public:
    CSRMatrix csr;
    void set_model(const vector<vector<vector<string>>>& corpus, double k = 1.5,
                   vector<double> b = {0.75, 0.75},
                   vector<double> w = {3.0, 1.0});
    vector<vector<double>> get_scores(const vector<vector<string>>& queries) override;
};
