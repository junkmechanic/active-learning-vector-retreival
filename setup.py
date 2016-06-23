from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="fast_similarity",
    ext_modules=cythonize("fast_similarity.pyx"),
)
