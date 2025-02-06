#pragma once

#ifndef BM25L_H
#define BM25L_H

# include "Base.h"

#include <string>
#include <vector>

using namespace std;

class BM25L : public Base {
public:
    double k;
    double b;
    double delta;

    void set_tf(double k, double b, double delta);
    void set_model(const vector<vector<string>>& corpus, double k = 1.5, double b = 0.75, double delta = 1.0);
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;

    ~BM25L() {}
};

#endif BM25L_H
