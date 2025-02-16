from setuptools import setup, Extension
import pybind11

sfc_module = Extension(
    'bm25_search',
    sources=[
        'model/Base.cpp',
        'model/TfIdf.cpp',
        'model/BM11.cpp',
        'model/BM15.cpp',
        'model/BM25.cpp',
        'model/BM25L.cpp',
        'model/BM25Plus.cpp',
        'model/BM25T.cpp',
        'binding.cpp'
        ],
    include_dirs=[pybind11.get_include()],
    language='c++'
)

setup(
    name='bm25_search',
    version='0.1.1',
    description='Python Package with BM25 Algorithms C++ extension using Pybind11',
    ext_modules=[sfc_module],
    zip_safe=False,
)