#pragma once

#include "Base.h"

// Small, shared storage/helper used by CSR adapters of algorithms that already
// calculate their exact TF values in the dense implementation.
struct CSRMatrix {
    vector<size_t> indptr;
    vector<size_t> indices;
    vector<double> data;
    unordered_map<string, size_t> vocabulary;
    vector<double> idf;

    void build(const unordered_map<string, vector<double>>& tf,
               const unordered_map<string, double>& idf_values,
               size_t doc_count);
    vector<vector<double>> scores(const vector<vector<string>>& queries) const;
};
