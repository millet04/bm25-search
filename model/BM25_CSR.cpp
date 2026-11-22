#include "BM25_CSR.h"

void BM25_CSR::set_model(const vector<vector<string>>& corpus, double k, double b) {
    this->k = k;
    this->b = b;
    init_csr(corpus);

    for (size_t document = 0; document < *doc_n; ++document) {
        for (const auto& entry : freq[document]) {
            const auto vocabulary_entry = csr_vocabulary.find(entry.first);
            if (vocabulary_entry == csr_vocabulary.end()) {
                continue;
            }
            const double term_frequency = static_cast<double>(entry.second);
            const double normalized_length = 1.0 - b + b * dl[document] / avgdl;
            const double value = term_frequency * (k + 1.0) /
                (term_frequency + k * normalized_length);
            add_csr_value(document, vocabulary_entry->second, value);
        }
        finish_csr_row();
    }
}
