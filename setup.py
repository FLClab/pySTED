
from distutils.command.build_py import build_py
from distutils.core import setup, Extension

import pysted

ext_modules = [Extension("pysted.cUtils", ["pysted/cUtils.c"])]

setup(name="pysted",
    version=pysted.__revision__,
    description="STED image simulator in Python",
    author=pysted.__author__,
    packages=["pysted"],
    platforms=["any"],
    license="LGPL",
    ext_modules = ext_modules,
    cmdclass = {"build_py": build_py}
)
