#pragma once

#ifndef BASE_H
#define BASE_H

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>

using namespace std;
namespace py = pybind11;

class Base {
public:
    size_t* doc_n;
    double avgdl = 0.0;

    vector<size_t> dl;
    unordered_map<string, double> df;
    unordered_map<string, double> idf;
    unordered_map<string, vector<double>> tf;
    vector<unordered_map<string, size_t>> freq;

    vector<vector<double>> get_scores(const vector<vector<string>>& queries);
    pair<vector<vector<double>>, vector<vector<size_t>>> get_topk(const vector<vector<string>>& queries, size_t n=5);
    vector<vector<string>> get_topk_docs(const vector<vector<string>>& queries, const vector<string>& docs, size_t n=5);

    void init(const vector<vector<string>>& corpus);
    void set_idf(void);
    void save_corpus(const string& filepath, const vector<string>& corpus);
    py::list load_corpus(const string& filepath);

    // Child classes that inherit from 'Base' should implement the following methods.
    virtual void set_tf(double k, double b);
    virtual void set_model(const vector<vector<string>>& corpus, double k, double b);
    virtual void save_model(const string& filepath);
    virtual void load_model(const string& filepath);

    Base() {
        doc_n = new size_t(0);
    }

    ~Base(){
        delete doc_n;
    }

};

#endif BASE_H

