Isotherm fitting module (:py:mod:`pyAPEP.isofit`)
====================================================

detailed discriptions are here.

Theory
-------
Here's theories.


Tutorials
------------------
For this tutorial on pyIAST, enter the `/test` directory of pyIAST. While you can type this code into the Python shell, I highly recommend instead opening a `Jupyter Notebook <http://jupyter.org/>`_.

First, import pyIAST into Python after installation.

.. code-block:: python

   import pyAPEP.isofit as isofit

For our tutorial, we have the pure-component methane and ethane adsorption isotherm data for metal-organic framework IRMOF-1 in Fig 1. We seek to predict the methane and ethane uptake in the presence of a binary mixture of methane and ethane in IRMOF-1 at the same temperature. As an example for this tutorial, we seek to predict the methane and ethane uptake of IRMOF-1 in the presence a 5/95 mol % ethane/methane mixture at a total pressure of 65.0 bar and 298 K.


Functions
-----------
