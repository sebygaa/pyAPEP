# pyAPEP Package Reference

**pyAPEP** (Python Adsorption Process Evaluation Package) is an all-in-one toolkit for evaluating adsorbent performance through process simulation. It comprises three main modules, each corresponding to a subpackage:

- **`pyAPEP.isofit`** – Isotherm fitting module for deriving isotherm models from data.
- **`pyAPEP.simide`** – Ideal PSA (Pressure Swing Adsorption) simulation module under idealized conditions.
- **`pyAPEP.simsep`** – Real PSA simulation module accounting for non-ideal effects (mass transfer, pressure drop, heat).

Below is a detailed reference for all public functions and classes in these subpackages, including their purpose and parameters.

---

## Table of Contents

1. [pyAPEP.isofit – Isotherm Fitting Module](#pyapepisofit--isotherm-fitting-module)
   - [best_isomodel(P, q, iso_par_nums, iso_fun_lists, iso_fun_index, tol)](#best_isomodel)
   - [fit_diffT(p_list, q_list, T_list, i_ref, iso_par_nums, iso_fun_lists, iso_fun_index, tol)](#fit_difft)
   - [IAST(isotherm_list, P_i, T)](#iastisotherm_list-p_i-t)
2. [pyAPEP.simide – Ideal PSA Simulation Module](#pyapepsimide--ideal-psa-simulation-module)
   - [IdealColumn (class)](#idealcolumn-class)
3. [pyAPEP.simsep – Real PSA Simulation Module](#pyapepsimsep--real-psa-simulation-module)
   - [column (class)](#column-class)
   - [Pressure Equalization Functions](#pressure-equalization-functions)
4. [Additional Information](#additional-information)

---

## pyAPEP.isofit – Isotherm Fitting Module

The **`isofit`** module is used to define adsorption isotherm functions from pressure–uptake data. It provides functions to automatically fit isotherm models to data, handle multi-temperature isotherm fitting, and predict mixture isotherms using Ideal Adsorbed Solution Theory (IAST).

### best_isomodel

```python
best_isomodel(P, q, iso_par_nums=[2, 3, 4], iso_fun_lists=None, iso_fun_index=None, tol=1e-05)
