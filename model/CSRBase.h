#pragma once

#include "Base.h"

class CSRBase : public Base {
public:
    // Document-major CSR representation of the calculated term-frequency matrix.
    vector<size_t> csr_indptr;
    vector<size_t> csr_indices;
    vector<double> csr_data;
    unordered_map<string, size_t> csr_vocabulary;
    vector<double> csr_idf;

    vector<vector<double>> get_scores(const vector<vector<string>>& queries) override;

public:
    void init_csr(const vector<vector<string>>& corpus);
    void add_csr_value(size_t document, size_t term, double value);
    void finish_csr_row();
};
