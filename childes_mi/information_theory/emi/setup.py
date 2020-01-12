from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os 

setup(name='_nij_op_cython',
      ext_modules=cythonize("_nij_op_cython.pyx"),
      include_dirs=[numpy.get_include()],
      include_path=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")],
      )

"""import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("emi", parent_package, top_path)
    libraries = []

    config.add_extension("_expected_mutual_info_fast",
                         sources=["_expected_mutual_info_fast.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries)


    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())"""