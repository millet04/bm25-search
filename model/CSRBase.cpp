#include "CSRBase.h"

#include <algorithm>

using namespace std;

void CSRBase::init_csr(const vector<vector<string>>& corpus) {
    init(corpus);
    set_idf();
    csr_vocabulary.clear();
    csr_idf.clear();
    for (const auto& entry : df) {
        const size_t index = csr_vocabulary.size();
        csr_vocabulary.emplace(entry.first, index);
        csr_idf.push_back(idf[entry.first]);
    }
    csr_indptr.clear();
    csr_indices.clear();
    csr_data.clear();
    csr_indptr.push_back(0);
}

void CSRBase::add_csr_value(size_t document, size_t term, double value) {
    if (document + 1 != csr_indptr.size()) {
        throw logic_error("CSR rows must be populated in document order.");
    }
    if (value != 0.0) {
        csr_indices.push_back(term);
        csr_data.push_back(value);
    }
}

void CSRBase::finish_csr_row() {
    csr_indptr.push_back(csr_indices.size());
}

vector<vector<double>> CSRBase::get_scores(const vector<vector<string>>& queries) {
    vector<vector<double>> scores;
    for (const auto& query : queries) {
        vector<double> score(*doc_n, 0.0);
        for (const auto& token : query) {
            const auto vocabulary_entry = csr_vocabulary.find(token);
            if (vocabulary_entry == csr_vocabulary.end()) {
                continue;
            }
            const size_t term = vocabulary_entry->second;
            for (size_t document = 0; document < *doc_n; ++document) {
                for (size_t offset = csr_indptr[document]; offset < csr_indptr[document + 1]; ++offset) {
                    if (csr_indices[offset] == term) {
                        score[document] += csr_data[offset] * csr_idf[term];
                        break;
                    }
                }
            }
        }
        scores.push_back(score);
    }
    return scores;
}
