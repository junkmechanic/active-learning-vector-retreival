from distutils.core import setup
# from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        "fast_similarity",
        ["fast_similarity.pyx"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="fast_similarity",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
