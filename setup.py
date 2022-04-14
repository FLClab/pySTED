
from distutils.command.build_py import build_py
from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

import pysted

# ext_modules = [Extension("pysted.cUtils", ["pysted/cUtils.c"]),
#                Extension("pysted._draw", ["pysted/_draw.c"]),
#                Extension("pysted.raster", ["pysted/raster.c"]),
#                Extension("pysted.bleach_funcs", ["pysted/bleach_funcs.c"])]

ext_modules = [Extension("pysted.cUtils", ["pysted/cUtils.c"]),
               Extension("pysted._draw", ["pysted/_draw.pyx"]),
               Extension("pysted.raster", ["pysted/raster.pyx"]),
               Extension("pysted.bleach_funcs", ["pysted/bleach_funcs.pyx"])]

setup(name="pysted",
      version=".".join((pysted.__version__, pysted.__revision__)),
      description="STED image simulator in Python",
      author=pysted.__author__,
      packages=["pysted"],
      platforms=["any"],
      license="LGPL",
      ext_modules = cythonize(ext_modules),
      cmdclass = {"build_py": build_py},
      include_dirs=[numpy.get_include()]
)
