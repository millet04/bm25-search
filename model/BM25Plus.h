#pragma once

#ifndef BM25PLUS_H
#define BM25PLUS_H

# include "Base.h"

#include <string>
#include <vector>

using namespace std;

class BM25Plus : public Base {
public:
    double k;
    double b;
    double delta;

    void set_tf(double k, double b, double delta);
    void set_model(const vector<vector<string>>& corpus, double k = 1.5, double b = 0.75, double delta = 1.0);
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;

    ~BM25Plus() {}
};

#endif BM25PLUS_H


