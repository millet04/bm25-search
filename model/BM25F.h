#pragma once

#ifndef BM25F_H
#define BM25F_H

# include "Base.h"

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

class BM25F : public Base {
public:
    double k;
    vector<double> b;
    vector<double> w;

    vector<vector<size_t>> field_dl;
    vector<double> field_avgdl;
  
    unordered_map<string, double> total_df;
    vector<unordered_map<string, double>> field_df;
    vector<vector<unordered_map<string, size_t>>> field_freq;
    
    void set_tf(size_t field_n, double k, vector<double> b, vector<double> w);
    void set_idf(void);
    void set_model(const vector<vector<vector<string>>>& corpus, double k = 1.5, vector<double> b = {0.75, 0.75}, vector<double> w = {3.0, 1.0});
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;
    
    vector<vector<unordered_map<string, string>>> get_topk_docs(const vector<vector<string>>& queries, const vector<unordered_map<string, string>>& docs, size_t n=5);

    ~BM25F() {}
};

#endif BM25F_H