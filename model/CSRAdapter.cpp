#include "CSRAdapter.h"

#include <algorithm>

void CSRMatrix::build(const unordered_map<string, vector<double>>& tf,
                      const unordered_map<string, double>& idf_values,
                      size_t doc_count) {
    indptr.assign(1, 0);
    indices.clear();
    data.clear();
    vocabulary.clear();
    idf.clear();

    vector<string> words;
    words.reserve(tf.size());
    for (const auto& item : tf) words.push_back(item.first);
    sort(words.begin(), words.end());
    for (size_t term = 0; term < words.size(); ++term) {
        vocabulary[words[term]] = term;
        auto it = idf_values.find(words[term]);
        idf.push_back(it == idf_values.end() ? 0.0 : it->second);
    }

    for (size_t doc = 0; doc < doc_count; ++doc) {
        for (size_t term = 0; term < words.size(); ++term) {
            const auto& values = tf.at(words[term]);
            const double value = doc < values.size() ? values[doc] : 0.0;
            if (value != 0.0) {
                indices.push_back(term);
                data.push_back(value);
            }
        }
        indptr.push_back(indices.size());
    }
}

vector<vector<double>> CSRMatrix::scores(const vector<vector<string>>& queries) const {
    const size_t doc_count = indptr.empty() ? 0 : indptr.size() - 1;
    vector<vector<double>> output;
    for (const auto& query : queries) {
        vector<double> score(doc_count, 0.0);
        for (const auto& token : query) {
            auto vocab_it = vocabulary.find(token);
            if (vocab_it == vocabulary.end()) continue;
            const size_t term = vocab_it->second;
            const double term_idf = term < idf.size() ? idf[term] : 0.0;
            for (size_t doc = 0; doc < doc_count; ++doc) {
                for (size_t p = indptr[doc]; p < indptr[doc + 1]; ++p) {
                    if (indices[p] == term) {
                        score[doc] += data[p] * term_idf;
                        break;
                    }
                }
            }
        }
        output.push_back(score);
    }
    return output;
}
