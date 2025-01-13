from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

#setup(
#    ext_modules = cythonize(Extension("mcts", ["mcts.pyx"], extra_compile_args=["-O2"]))
#)

setup(
    ext_modules = cythonize(["mcts.pyx",])
)
