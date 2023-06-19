pyAPEP
========

pyAPEP consists of three modules: :py:mod:`isofit`, :py:mod:`simide`, and :py:mod:`simsep`. First, :py:mod:`isofit` module automatically derives the isotherm function from the given data. Additionally, the module includes a function that derives isotherm functions under different temperature conditions and that of mixture gas using IAST. Next, using :py:mod:`simide` module that assumes an ideal PSA process, users can derive the theoretical maximum performance of adsorbent with a relatively simple calculation. Finally, :py:mod:`simsep` module provides a function that simulates the actual PSA process through detailed calculation and performs dynamic behavior and breakthrough test.


.. image:: images/pyAPEP_structure.png
  :width: 1000
  :alt: pyAPEP_structure
  :align: center

-------------

Usages, the function structure within the module, and the theory used are summarized for each module.


.. toctree::
   :maxdepth: 2
   :caption: Modules:

   isofit
   simide
   simsep


----------------------------------------------------------------------------------------------------------------------------------------------------------------------