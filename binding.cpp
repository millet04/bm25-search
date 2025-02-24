#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "model/Base.h"
#include "model/TfIdf.h"
#include "model/BM25.h"
#include "model/BM11.h"
#include "model/BM15.h"
#include "model/BM25L.h"
#include "model/BM25Plus.h"
#include "model/BM25T.h"
#include "model/BM25F.h"


using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(bm25_search, m) {
    py::class_<Base>(m, "Base")
        .def(py::init<>()) 
        .def("init", &Base::init, "Initialize the corpus", py::arg("corpus"))
        .def("set_idf", &Base::set_idf, "Calculate the IDF values")
        .def("get_scores", &Base::get_scores, "Get scores for queries", py::arg("queries"))
        .def("get_topk", &Base::get_topk, "Get top-k scores and indices", py::arg("queries"), py::arg("n")=5)
        .def("get_topk_docs", &Base::get_topk_docs, "Get top-k documents", py::arg("queries"), py::arg("docs"), py::arg("n")=5)
        .def("save_corpus", &Base::save_corpus, "Save a corpus", py::arg("filepath"), py::arg("corpus"))
        .def("load_corpus", &Base::load_corpus, "Save a corpus", py::arg("filepath"))
        .def("set_tf", &Base::set_tf, "Set term frequency values", py::arg("k"), py::arg("b"))
        .def("set_model", &Base::set_model, "Set a model", py::arg("corpus"), py::arg("k"), py::arg("b"))
        .def("save_model", &Base::save_model, "Save a model", py::arg("filepath"))
        .def("load_model", &Base::load_model, "Load a model", py::arg("filepath"))
        .def_readwrite("doc_n", &Base::doc_n)
        .def_readwrite("avgdl", &Base::avgdl)
        .def_readwrite("dl", &Base::dl)
        .def_readwrite("df", &Base::df)
        .def_readwrite("idf", &Base::idf)
        .def_readwrite("tf", &Base::tf)
        .def_readwrite("freq", &Base::freq);

    py::class_<TFIDF, Base>(m, "TFIDF")
        .def(py::init<>())
        .def("set_tf", &TFIDF::set_tf, "Set term frequency values")
        .def("set_idf", &TFIDF::set_idf, "Calculate the IDF values")
        .def("set_model", &TFIDF::set_model, "Set a TF-IDF model", py::arg("corpus"))
        .def("save_model", &TFIDF::save_model, "Save a TF-IDF model", py::arg("filepath"))
        .def("load_model", &TFIDF::load_model, "Load a TF-IDF model", py::arg("filepath"));

    py::class_<BM25, Base>(m, "BM25")
        .def(py::init<>())
        .def("set_tf", &BM25::set_tf, "Set term frequency values", py::arg("k"), py::arg("b"))
        .def("set_model", &BM25::set_model, "Set a BM25 model", py::arg("corpus"), py::arg("k")=1.5, py::arg("b")=0.75)
        .def("save_model", &BM25::save_model, "Save a BM25 model", py::arg("filepath"))
        .def("load_model", &BM25::load_model, "Load a BM25 model", py::arg("filepath"))
        .def_readwrite("k", &BM25::k)
        .def_readwrite("b", &BM25::b);

    py::class_<BM11, Base>(m, "BM11")
        .def(py::init<>())
        .def("set_tf", &BM11::set_tf, "Set term frequency values", py::arg("k"))
        .def("set_model", &BM11::set_model, "Set a BM11 model", py::arg("corpus"), py::arg("k") = 1.5)
        .def("save_model", &BM11::save_model, "Save a BM11 model", py::arg("filepath"))
        .def("load_model", &BM11::load_model, "Load a BM11 model", py::arg("filepath"))
        .def_readwrite("k", &BM11::k);

    py::class_<BM15, Base>(m, "BM15")
        .def(py::init<>())
        .def("set_tf", &BM15::set_tf, "Set term frequency values", py::arg("k"))
        .def("set_model", &BM15::set_model, "Set a BM15 model", py::arg("corpus"), py::arg("k") = 1.5)
        .def("save_model", &BM15::save_model, "Save a BM15 model", py::arg("filepath"))
        .def("load_model", &BM15::load_model, "Load a BM15 model", py::arg("filepath"))
        .def_readwrite("k", &BM15::k);

    py::class_<BM25L, Base>(m, "BM25L")
        .def(py::init<>())
        .def("set_tf", &BM25L::set_tf, "Set term frequency values", py::arg("k"), py::arg("b"), py::arg("delta"))
        .def("set_model", &BM25L::set_model, "Set a BM25L model", py::arg("corpus"), py::arg("k") = 1.5, py::arg("b") = 0.75, py::arg("delta")=1.0)
        .def("save_model", &BM25L::save_model, "Save a BM25L model", py::arg("filepath"))
        .def("load_model", &BM25L::load_model, "Load a BM25L model", py::arg("filepath"))
        .def_readwrite("k", &BM25L::k)
        .def_readwrite("b", &BM25L::b)
        .def_readwrite("delta", &BM25L::delta);

    py::class_<BM25Plus, Base>(m, "BM25Plus")
        .def(py::init<>())
        .def("set_tf", &BM25Plus::set_tf, "Set term frequency values", py::arg("k"), py::arg("b"), py::arg("delta"))
        .def("set_model", &BM25Plus::set_model, "Set a BM25Plus model", py::arg("corpus"), py::arg("k") = 1.5, py::arg("b") = 0.75, py::arg("delta") = 1.0)
        .def("save_model", &BM25Plus::save_model, "Save a BM25Plus model", py::arg("filepath"))
        .def("load_model", &BM25Plus::load_model, "Load a BM25Plus model", py::arg("filepath"))
        .def_readwrite("k", &BM25Plus::k)
        .def_readwrite("b", &BM25Plus::b)
        .def_readwrite("delta", &BM25Plus::delta);

    py::class_<BM25T, Base>(m, "BM25T")
        .def(py::init<>())
        .def("set_tf", &BM25T::set_tf, "Set term frequency values", py::arg("k"), py::arg("b"), py::arg("eps"), py::arg("max_iter"))
        .def("compute_optimal_k", &BM25T::compute_optimal_k, "Compute the optimal k", py::arg("word"), py::arg("k"), py::arg("sum_log_c"), py::arg("eps"), py::arg("max_iter"))
        .def("set_model", &BM25T::set_model, "Set a BM25T model", py::arg("corpus"), py::arg("k") = 1.5, py::arg("b") = 0.75, py::arg("eps") = 0.05, py::arg("max_iter") = 100)
        .def("save_model", &BM25T::save_model, "Save a BM25T model", py::arg("filepath"))
        .def("load_model", &BM25T::load_model, "Load a BM25T model", py::arg("filepath"))
        .def_readwrite("k", &BM25T::k)
        .def_readwrite("b", &BM25T::b)
        .def_readwrite("eps", &BM25T::eps)
        .def_readwrite("max_iter", &BM25T::max_iter)
        .def_readwrite("optk_set", &BM25T::optk_set);

    py::class_<BM25F, Base>(m, "BM25F")
        .def(py::init<>())
        .def("set_tf", &BM25F::set_tf, "Set term frequency values", py::arg("field_n"), py::arg("k"), py::arg("b"), py::arg("w"))
        .def("set_idf", &BM25F::set_idf, "Calculate the IDF values")
        .def("set_model", &BM25F::set_model, "Set a BM25F model", py::arg("corpus"), py::arg("k") = 1.5, py::arg("b") = vector<double>{ 0.75, 0.75 }, py::arg("w") = vector<double>{ 3.0, 1.0 })
        .def("save_model", &BM25F::save_model, "Save a BM25F model", py::arg("filepath"))
        .def("load_model", &BM25F::load_model, "Load a BM25F model", py::arg("filepath"))
        .def("get_topk_docs", &BM25F::get_topk_docs, "Get top-k documents", py::arg("queries"), py::arg("docs"), py::arg("n") = 5)
        .def_readwrite("k", &BM25F::k)
        .def_readwrite("b", &BM25F::b)
        .def_readwrite("w", &BM25F::w)
        .def_readwrite("field_avgdl", &BM25F::field_avgdl)
        .def_readwrite("field_dl", &BM25F::field_dl)
        .def_readwrite("total_df", &BM25F::total_df)
        .def_readwrite("field_freq", &BM25F::field_freq);
}