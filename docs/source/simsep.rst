Real PSA simulation module (:py:mod:`pyAPEP.simsep`)
=======================================================
This module enables ideal PSA simulation using isotherm function and operating conditions.

First, import simsep into Python after installation.

.. code-block:: python

   import pyAPEP.simsep as simsep

Then users need to 10-steps to simulate.
    1. Isotherm selection
    2. Column design
    3. Adsorbent information
    4. Gas property information 
    5. Mass transfer information
    6. Thermal information 
    7. Boundary condition setting
    8. Initial condition setting 
    9. Simulation run
    10. Graph

In next section, detailed steps are explained.

------------------------------------------------------

Usage
-------

1. Isotherm selection
''''''''''''''''''''''''''''''''''''''''''''''

Here, we use extended Langmuir isotherm as an example.

.. math::

    q_{i} = \frac{q_{m,i}b_{i}P_{i}}{1+\sum^{n}_{j=1}b_{j}P_{j}}

First, we need to import some libraries.

.. code-block:: python

    import numpy as np 
    import matplotlib.pyplot as plt

Then, define some parameters of the extended Langmuir isotherm.

.. code-block:: python

    # Define parameters of extended Langmuir isotherm 
    qm1 = 1
    qm2 = 0.1
    b1 = 0.5
    b2 = 0.05

    dH1 = 10000     # J/mol
    dH2 = 15000     # J/mol
    R_gas = 8.3145 # J/mol/K
    T_ref = 300    # K

The isotherm is defined as follows.
    
.. code-block:: python

    # Define the isotherm 
    def extLang(P, T):
        P1 = P[0]*np.exp(dH1/R_gas*(1/T-1/T_ref))
        P2 = P[1]*np.exp(dH1/R_gas*(1/T-1/T_ref))
        deno = 1 + b1*P1 + b2*P2
        q1 = qm1*b1*P1 / deno
        q2 = qm2*b2*P2 / deno
        return q1, q2

2. Column design
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    # Set the design parameter of a column
    col_len         = 1      # (m)                                  
    cross_sect_area = 0.0314 # (m^2)                                    
    num_comp        = 2
    node = 41                # The number of nodes

    # Column definition
    Column1 = simsep.column(col_len, cross_sect_area, num_comp, N_node= node)

3. Adsorbent information
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    voidfrac = 0.4                                          # Void fraction
    rho      = 1100                                         # Solid density (kg/m^2)
    Column1.adsorbent_info(extLang, voidfrac, rho_s = rho)  # Adsorbent information
    print(Column1)                                          # Check 

4. Gas property information 
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    Mmol = [0.032, 0.044]   # Molecular weights of gases (kg/mol)                    
    visc = [0.01, 0.01]     # Viscosities of gases (Pa sec)
    
    Column1.gas_prop_info(Mmol, visc) # Gas property information
    print(Column1)                    # Check 

5. Mass transfer information 
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    MTC = [0.05, 0.05]      # Mass transfer coefficients                    
    a_surf = 400            # Volumatric specific surface area (m2/m3)
    
    Column1.mass_trans_info(MTC, a_surf) # Mass transfer information
    print(Column1)                       # Check

6. Thermal information 
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    dH_ads = [10000,16000]  # Heat of adsorption (J/mol)                    
    Cp_s = 5                # Solid heat capacity (J/kg K)
    Cp_g = [10,10]          # Gas heat capacity (J/mol K)
    h_heat = 10             # Heat transfer coefficient between solid and gas (J/m^2 K s)
    
    Column1.thermal_info(dH_ads, Cp_s, Cp_g, h_heat) # Mass transfer information
    print(Column1)                                   # Check

7. Boundary condition setting 
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    P_inlet  = 9              # Inlet pressure  (bar)                 
    P_outlet = 8.0            # Outlet pressure (bar)
    T_feed   = 300            # Feed in temperature (K)
    y_feed = [0.5,0.5]        # Feed composition (mol/mol)
    
    Column1.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed) # Boundary condition
    print(Column1)                                            # Check

8. Initial condition setting 
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    P_init = 8*np.ones(node)                       # Initial pressure (bar)                
    y_init = [0.2*np.ones(node), 0.8*np.ones(node)]  # Gas phase mol fraction (mol/mol)
    Tg_init = 300*np.ones(node)                    # Initial gas temperature (K)
    Ts_init = 300*np.ones(node)                    # Initial solid temperature (K)
    
    P_partial = [P_init*y_init[0], P_init*y_init[1]] # Partial pressure (bar)
    q_init = extLang(P_partial, Ts_init)             # Solid phase uptake (mol/kg)
    
    Column1.initialC_info(P_init, Tg_init, Ts_init, y_init, q_init) # Initial condition
    print(Column1)                                                  # Check

9. Simulation run
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    y, z, t = Column1.run_mamoen(2000, n_sec=10, CPUtime_print=True)
    
10. Graph
''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    # Simulation result: Pressure according to z-domain
    Column1.Graph_P(200,loc = [1.2, 0.9]) 
    # Simulation result: Gas concentration according to z-domain
    Column1.Graph(200, 0, yaxis_label=
    'Gas Concentration (mol/m$^3$)',loc = [1.2, 0.9])    
    # Simulation result: Solid concentration according to z-domain
    Column1.Graph(200, 2, yaxis_label=
    'Soild concentration (uptake) (mol/kg)',loc = [1.2, 0.9])

The results are shown in Fig. (a)--(c)

.. figure:: images/simsep_res2.png
   :alt: simsep graph 1
   :figwidth: 60%
   :align: center

----------------------------------------

Class documentation
----------------------------------

.. automodule:: pyAPEP.simsep
    :special-members:
    :members:
    :member-order: bysource

---------------------------------

Theory
-------


Mass balance
'''''''''''''''''''''''''

The mass balance relationship, shown below, is used to describe the pressure swing adsorption process.

.. math::

    \frac{\partial C_{i}}{\partial t} = -\frac{\partial (uC_{i})}{\partial z} + D_{dis} \frac{\partial^2 C_{i}}{\partial z^2} - \rho_{s}\frac{(1-\epsilon)}{\epsilon}\frac{\partial q_{i}}{\partial z_{i}}

where

    * :math:`C_{i} =` Concentration of component i :math:`(mol/m^3)`
    * :math:`t =` Time :math:`(s)`
    * :math:`D_{dis} =` Dispersion coefficient :math:`(m^2/s)`
    * :math:`z =` Length in axial direction :math:`(m)`
    * :math:`\epsilon =` Void fraction :math:`(m^3/m^3)`
    * :math:`\rho_{s} =` Density of solid :math:`(kg/m^3)`
    * :math:`q_{i} =` Uptake of component i :math:`(mol/kg)`


`Momentum balance <http://dns2.asia.edu.tw/~ysho/YSHO-English/2000%20Engineering/PDF/Che%20Eng%20Pro48,%2089.pdf>`_ 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

In the adsorption process simulation, the pressure drop :math:`\Delta P` should be considered to solve the momentum balance along the length of the bed. In 1952, Ergun proposed an equation to consider :math:`\Delta P` in a packed bed, and the same equation was used to solve the momentum balance in this package, similar to many previous studies.

The Ergun equation represents the relationship between the pressure drop and resultant fluid flow in packed beds, as shown below. The equation was derived from experimental measurements and theoretical postulates. 

.. math::

    - \frac{\partial P}{\partial z} = \frac{180 \mu }{d_{p}^2 } \frac{(1 - \varepsilon)^2}{\varepsilon^3} u + \frac{7}{4} \frac{\rho_{f}}{d_{P}} \frac{1 - \varepsilon}{\varepsilon^3} u|{u}|

where

    * :math:`\partial P =` Pressure drop :math:`(bar)`
    * :math:`L =` Height of the bed :math:`(m)`
    * :math:`\mu =` Fluid viscosity :math:`(Pa \cdot s)`
    * :math:`\varepsilon =` Void space of the bed
    * :math:`u =` Fluid superficial velocity :math:`(m/s)`
    * :math:`d_{P} =` Particle diameter :math:`(m)`
    * :math:`\rho_{f} =` Density of Fluid :math:`(kg/m^3)`

    
`Energy balance <https://doi.org/10.1016/j.compchemeng.2016.11.021>`_ 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The energy balance, shown below, is also used to describe the pressure swing adsorption process.

.. math::

    Convection = -CC_{p,g}u\frac{\partial T_{g}}{\partial z} \frac{h}{\varepsilon}a_{surf}(T_{s}-T_{g})

.. math::
    Conduction =  \lambda_{cond} \frac{\partial^{2} T_{s}}{\partial z^{2}}

.. math::
    Generation = \sum_{i=1}^{n}(-\Delta H_{ad,i})\rho_{s} (1-\varepsilon) \frac{\partial q_{i}}{\partial t}

**The overall energy balance under the boundary condition for the gas phase:**

.. math::
    \sum_{i=1}^{n}C_{i}C_{p,i}\frac{\partial T_{g}}{\partial t} 
    = h a_{surf} (T_{s} - T_{g})
    -\sum_{i=1}^{n}uC_{i}C_{p,i}\frac{\partial T_{g}}{\partial z}

**The overall energy balance under the boundary condition for the solid phase:**

.. math::
    \left( (1-\varepsilon) \rho_{s} \sum_{i=1}^{n} (q_{i} C_{p,i}) + \rho_{s}C_{p,s}  \right) &\frac{\partial T_{s}}{\partial t}
    = h a_{surf}(T_{g} - T_{s}) + \lambda_{cond} \frac{\partial^{2} T_{s}}{\partial z^{2}}\\  
    +&\sum_{i=1}^{n}(-\Delta H_{ad,i})\rho_{s} (1-\varepsilon) \frac{\partial q_{i}}{\partial t}

where

    * :math:`C =` Concentration :math:`(mol/m^3)`
    * :math:`C_{p, g} =` Gas heat capacity :math:`(J/mol \cdot K)`
    * :math:`T_{g} =` Gas temperature :math:`(K)`
    * :math:`t =` Time :math:`(s)`
    * :math:`u =` Velocity :math:`(m/s)`
    * :math:`z =` Length in axial direction :math:`(m)`
    * :math:`h =` Enthalpy :math:`(J/m^2 \cdot K \cdot s)`
    * :math:`\varepsilon =` Void fraction :math:`(m^3/m^3)`
    * :math:`a_{surf} =` Interfacial area concentration :math:`(m^2/m^3)`
    * :math:`T_{s} =` Solid temperature :math:`(K)`
    * :math:`\rho_{s} =` Density of solid :math:`(kg/m^3)`
    * :math:`q_{i} =` Uptake of component i :math:`(mol/kg)`




--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
