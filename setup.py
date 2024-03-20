
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options

INCLUDE_DIRS = []

try:
    import numpy

    INCLUDE_DIRS.append(numpy.get_include())
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


ext_modules = [
    Extension("pysted.cUtils", ["pysted/cUtils.c"], include_dirs=INCLUDE_DIRS),
    Extension("pysted._draw", ["pysted/_draw.pyx"], include_dirs=INCLUDE_DIRS),
    Extension("pysted.raster", ["pysted/raster.pyx"], include_dirs=INCLUDE_DIRS),
    Extension("pysted.bleach_funcs", ["pysted/bleach_funcs.pyx"], include_dirs=INCLUDE_DIRS)
]
for ext_module in ext_modules:
    ext_module.cython_directives = {"embedsignature": True}

Options.cimport_from_pyx = False

setup(
    ext_modules = cythonize(ext_modules),
)
