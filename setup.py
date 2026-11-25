from setuptools import setup, Extension
import pybind11
from pathlib import Path

long_description = Path('README.md').read_text(encoding='utf-8')

sfc_module = Extension(
    'bm25_search',
    sources=[
        'model/Base.cpp',
        'model/CSRBase.cpp',
        'model/CSRAdapter.cpp',
        'model/TfIdf.cpp',
        'model/BM11.cpp',
        'model/BM15.cpp',
        'model/BM25.cpp',
        'model/BM25_CSR.cpp',
        'model/CSRAlgorithms.cpp',
        'model/BM25L.cpp',
        'model/BM25Plus.cpp',
        'model/BM25T.cpp',
        'model/BM25F.cpp',
        'model/BM25T_CSR.cpp',
        'model/BM25F_CSR.cpp',
        'binding.cpp'
        ],
    include_dirs=[pybind11.get_include()],
    language='c++'
)

setup(
    name='bm25_search',
    version='1.0.0',
    description='Python Package with BM25 Algorithms C++ extension using Pybind11',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    ext_modules=[sfc_module],
    zip_safe=False,
)
