#pragma once

#ifndef BM11_H
#define BM11_H

# include "Base.h"

#include <string>
#include <vector>

using namespace std;

class BM11 : public Base {
public:
    double k;

    void set_tf(double k);
    void set_model(const vector<vector<string>>& corpus, double k = 1.5);
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;

    ~BM11() {}
};

#endif BM11_H