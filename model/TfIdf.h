#pragma once

#ifndef TFIDF_H
#define TFIDF_H

# include "Base.h"

#include <string>
#include <vector>

using namespace std;

class TFIDF: public Base {
public:
    void set_tf(void);
    void set_idf(void);
    void set_model(const vector<vector<string>>& corpus);
    void save_model(const string& filepath) override;
    void load_model(const string& filepath) override;

    ~TFIDF() {}
};

#endif TFIDF_H
