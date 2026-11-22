#pragma once
#include "CSRBase.h"

#define DECLARE_CSR_ALGORITHM(Name) \
class Name##_CSR : public CSRBase { public: double k; double b; double delta; void set_model(const vector<vector<string>>& corpus, double k=1.5, double b=0.75, double delta=1.0); };
DECLARE_CSR_ALGORITHM(BM11)
DECLARE_CSR_ALGORITHM(BM15)
DECLARE_CSR_ALGORITHM(BM25L)
DECLARE_CSR_ALGORITHM(BM25Plus)
class TFIDF_CSR : public CSRBase { public: void set_model(const vector<vector<string>>& corpus); };
