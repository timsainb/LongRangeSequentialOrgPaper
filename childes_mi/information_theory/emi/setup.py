from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os 

setup(name='_nij_op_cython',
      ext_modules=cythonize("_nij_op_cython.pyx"),
      include_dirs=[numpy.get_include()],
      include_path=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")],
      )


setup(name='_expected_mutual_info_fast',
      ext_modules=cythonize("_expected_mutual_info_fast.pyx"),
      include_dirs=[numpy.get_include()],
      include_path=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")],
      )


setup(name='_expected_mutual_info_fast_sklearn',
      ext_modules=cythonize("_expected_mutual_info_fast_sklearn.pyx"),
      include_dirs=[numpy.get_include()],
      include_path=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")],
      )
