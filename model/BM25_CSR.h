#pragma once

#include "CSRBase.h"

class BM25_CSR : public CSRBase {
public:
    double k;
    double b;

    void set_model(const vector<vector<string>>& corpus, double k = 1.5, double b = 0.75);
};
