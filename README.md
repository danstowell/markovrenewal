markovrenewal - code for multiple Markov renewal process inference
=============

Given:

1. a set of input data where each data-point has a "state" and time,
2. a generative model for transitions from statetime to statetime (a *Markov renewal process* model),

this code performs inference to determine the most likely clustering of the 
data into individual MRP streams, plus noise.

For detailed description see this paper:

* Stowell & Plumbley (2013),
  ``Segregating event streams and noise with a Markov renewal process model''
  accepted in Journal of Machine Learning Research
  http://arxiv.org/abs/1211.2972


System requirements
-------------------

* Python 2.7 or later
* Cython  (NOT strictly required - the pure python code can be used as-is, but the "setup.py" file I provide will attempt to compile it to binary code using Cython. It runs faster that way, so it is strongly recommended.)

* Optional: Python modules numpy, scipy, matplotlib (NOT required for the core code, only for the included experiment scripts)
* Optional: In order to run the birdsong experiment script, you need various POSIX command-line audio tools such as sox, lame
* Optional: graphviz, in order to generate graphviz plots of inferred graphs

Usage
-----

To install the main code module, run

     python setup.py build_ext
     sudo python setup.py install

Then to get started using the code, from within a Python session:

     import markovrenewal
     help(markovrenewal)

You may find it more helpful to look at usage as embodied in the scripts inside the "experiments" folder.
In that folder is also a README which tells you how the scripts correspond to the figures in the research journal article cited above.


Licence
-------

This code was written by Dan Stowell 2012--2013.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


