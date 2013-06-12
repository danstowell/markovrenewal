'''
python setup.py build_ext
sudo python setup.py install

#or if you want it to be built in-place:
python setup.py build_ext --inplace
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("markovrenewal", ["markovrenewal.py"])]

setup(
  name = 'markovrenewal',
  description = 'Multiple Markov renewal process inference',
  version = '1.0',
  author = 'Dan Stowell',
  license = 'GPL-2+',
  url = 'https://code.soundsoftware.ac.uk/projects/markovrenewal',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

