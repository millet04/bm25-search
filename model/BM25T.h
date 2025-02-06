#pragma once

#ifndef BM25T_H
#define BM25T_H

# include "Base.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

class BM25T : public Base {
public:
    double k;
    double b;
    double eps;
    unsigned int max_iter;
    unordered_map<string, double> optk_set;

    void set_tf(double k, double b, double eps, unsigned int max_iter);
    double compute_optimal_k(const string& word, double k, double sum_log_c, double eps, unsigned int max_iter);
    void set_model(const vector<vector<string>>& corpus, double k = 1.5, double b = 0.75, double eps = 0.1, unsigned int max_iter = 100);
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;

    ~BM25T() {}
};

#endif BM25T_H
