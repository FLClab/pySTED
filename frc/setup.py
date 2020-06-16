
from distutils.core import Extension, setup
from Cython.Build import cythonize

ext = Extension(name="fourierRingCorrelation", sources=["fourierRingCorrelation.pyx"])
setup(ext_modules=cythonize(ext))
