
from distutils.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

import numpy

import pysted

class build_ext_openmp(build_ext):
    # https://www.openmp.org/resources/openmp-compilers-tools/
    # python setup.py build_ext --help-compiler
    openmp_compile_args = {
        'msvc':  [['/openmp']],
        'intel': [['-qopenmp']],
        '*':     [['-fopenmp'], ['-Xpreprocessor','-fopenmp']],
    }
    openmp_link_args = openmp_compile_args # ?

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith('intel'):
            compiler = 'intel'
        if compiler not in self.openmp_compile_args:
            compiler = '*'

        # thanks to @jaimergp (https://github.com/conda-forge/staged-recipes/pull/17766)
        # issue: qhull has a mix of c and c++ source files
        #        gcc warns about passing -std=c++11 for c files, but clang errors out
        compile_original = self.compiler._compile
        def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # remove c++ specific (extra) options for c files
            if src.lower().endswith('.c'):
                extra_postargs = [arg for arg in extra_postargs if not arg.lower().startswith('-std')]
            return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # monkey patch the _compile method
        self.compiler._compile = compile_patched

        # store original args
        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args    = list(ext.extra_link_args)

        # try compiler-specific flag(s) to enable openmp
        for compile_args, link_args in zip(self.openmp_compile_args[compiler], self.openmp_link_args[compiler]):
            try:
                ext.extra_compile_args = _extra_compile_args + compile_args
                ext.extra_link_args    = _extra_link_args    + link_args
                return super(build_ext_openmp, self).build_extension(ext)
            except:
                print(f">>> compiling with '{' '.join(compile_args)}' failed")

        print('>>> compiling with OpenMP support failed, re-trying without')
        ext.extra_compile_args = _extra_compile_args
        ext.extra_link_args    = _extra_link_args
        return super(build_ext_openmp, self).build_extension(ext)

# ext_modules = [Extension("pysted.cUtils", ["pysted/cUtils.c"]),
#                Extension("pysted._draw", ["pysted/_draw.c"]),
#                Extension("pysted.raster", ["pysted/raster.c"]),
#                Extension("pysted.bleach_funcs", ["pysted/bleach_funcs.c"])]

ext_modules = [Extension("pysted.cUtils", ["pysted/cUtils.c"]),
               Extension("pysted._draw", ["pysted/_draw.pyx"]),
               Extension("pysted.raster", ["pysted/raster.pyx"]),
               Extension(
                "pysted.bleach_funcs",
                ["pysted/bleach_funcs.pyx"]
               )]

setup(name="pysted",
      version=".".join((pysted.__version__, pysted.__revision__)),
      description="STED image simulator in Python",
      author=pysted.__author__,
      packages=["pysted"],
      platforms=["any"],
      license="LGPL",
      ext_modules = ext_modules,
      cmdclass = {
        "build_py": build_py,"build_ext" : build_ext_openmp
      },
      include_dirs=[numpy.get_include()],
      install_requires=[
        "certifi",
        "cycler",
        "Cython",
        "decorator",
        "imageio",
        "kiwisolver",
        "matplotlib",
        "networkx",
        "numpy",
        "pyparsing",
        "python-dateutil",
        "PyWavelets",
        "scikit-image",
        "scipy",
        "six",
        "tifffile",
        "tqdm",
        "pyqt5",
      ]
)
