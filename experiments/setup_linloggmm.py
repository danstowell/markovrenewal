'''
python setup_linloggmm.py build_ext --inplace
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("linloggmm", ["linloggmm.py"])]

setup(
  name = 'Markov Renewal linloggmm cython',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

