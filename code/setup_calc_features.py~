from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


extensions = [
    Extension("ensemble_cython", ["ensemble_cython.pyx"], language="c++",
        extra_compile_args = ["-O3", "-Wno-unused-function", "-Wno-unneeded-internal-declaration"]),
]

setup(

    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(
        extensions
    ),
)
