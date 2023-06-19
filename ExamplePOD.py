 # %%
from simsep import *
# %%
N = 501
A_cros = 0.1
L = 0.2
c1 = column(L,A_cros, n_component = 2,N_node = N)

## Adsorbent
isopar1 = [0.1, 1]
isopar2 = [0.1, 0.1]
def iso_fn_test(P,T):
    b1 = isopar1[1]*np.exp(30E3/8.3145*(1/T-1/300))
    b2 = isopar2[1]*np.exp(20E3/8.3145*(1/T-1/300))
    #denom = 1 + b1*P[0]+ b2*P[1]
    numor0 = isopar1[0]*b1*P[0]
    numor1 = isopar2[0]*b2*P[1]
    q_return = [numor0, numor1]
    return q_return

# %%
epsi_test = 0.4         # macroscopic void fraction (m^3/m^3)
D_particle_dia = 0.01   # particle diameter (m)
rho_s_test = 1000       # solid density (kg/m^3)
c1.adsorbent_info(iso_fn_test,epsi_test,D_particle_dia, rho_s_test)

M_m_test  = [0.044, 0.028]      ## molar mass    (kg/mol)
mu_test = [16.13E-6, 16.13E-6]    ## gas viscosity (Pa sec)

c1.gas_prop_info(M_m_test, mu_test)

## Mass transfer coefficients
D_dis_test = [1E-6, 1E-6]   # m^2/sec
k_MTC = [1E-2, 1E-2]        # m/sec
a_surf = 50                 # m^2/m^3
c1.mass_trans_info(k_MTC, a_surf, D_dis_test)

## Thermal properties
Del_H = [31.164e3,20.856e3] # J/mol
Cp_s = 935                  # J/kg/K
Cp_g = [38.236, 35.8]       # J/mol/K
h_heat = 100                # J/sec/m^2/K
c1.thermal_info(Del_H,Cp_s,Cp_g,h_heat,)

## Boundary condition
Pin_test = 2.0      # inlet pressure (bar)
yin_test = [1, 0]   # inlet composition (mol/mol)
Tin_test = 300      # inlet temperature (K)
Q_in_test = 0.02*A_cros*epsi_test  # volumetric flowrate (m^3/sec)
Cvin_test = 1E-1    # inlet valve constant (m/sec/bar)

Pout_test = 2       # outlet pressure (bar)
Cvout_test = 2E-2   # outlet valve constant (m/sec/bar)
c1.boundaryC_info(Pout_test,Pin_test,Tin_test,yin_test,
                  Cvin_test,Cvout_test,Q_in_test,assigned_v_option = True)

# %%
## Initial condition
P_init = 2.0*np.ones(N)
Tg_init = 300*np.ones(N)
Ts_init = 300*np.ones(N)
y_init = [0.0*np.ones(N), 1.0*np.ones(N)]

q1_init, q2_init = iso_fn_test([0.0*P_init, P_init], Tg_init,)
q_init = [q1_init, q2_init]
# %%
c1.initialC_info(P_init, Tg_init, Ts_init, y_init, q_init)
print(c1)

# %%
# find basis without running simulation
c1.find_basis(True, show_n = 40)

# %%
# Run mamoen and copy
y_res, z_dom, t_dom = c1.run_ma(10, 20, CPUtime_print = True)
c1_POD = c1.copy()

# %%
# Find basis again
U, Sig, Vt, Fig_sig = c1.find_basis(True, show_n = 10)
#U, Sig, Vt = c1.find_basis(False, show_n = 40)

# %%
y_POD, z_dom, t_dom = c1_POD.run_ma_POD(10, 20, N_basis= 40, 
                                        CPUtime_print = True, )

# %%
# Visualization of the simulation results
c1.Graph(1, 0, 
         yaxis_label = 'Concentration (mol/m$^{3}$)\n of component 1', 
         dpi = 100, loc = [0.9,0.87])
c1_POD.Graph(1, 0, 
         yaxis_label = 'Concentration (mol/m$^{3}$)\n of component 1', 
         dpi = 100, loc = [0.9,0.87])

# %%
# Visualization: Solid phase concentration
c1.Graph(1, 2, 
         yaxis_label = 'Uptake (mol/kg)\n of component 1', 
         dpi = 100, loc = [0.9,0.87])
c1_POD.Graph(1, 2, 
         yaxis_label = 'Uptake (mol/kg)\n of component 1', 
         dpi = 100, loc = [0.9,0.87])

# %%
# CODE
x_lab = [0.8, 2.7]
t_CPU = np.array([0.328, 0.011])*60
plt.bar(x_lab, t_CPU, color = 'k', zorder = 2)
plt.xlim([-0.5,4.5])
plt.yscale('log')

plt.ylabel("CPU time (sec)", fontsize = 13.5)
plt.xticks(x_lab, 
           ["Full \nsimulation", "Simulation \nwith POD method"],
           fontsize = 13.5)
plt.grid(linestyle=":", which = "both", linewidth = 0.5, color = 'gray')

minor_yticks = np.logspace(-2, 2, num=9, base=10)  # Example: Logarithmic minor ticks from 0.01 to 100

plt.yticks(minor_yticks,fontsize = 12.5)
plt.ylim([0.1, 40])
print(t_CPU)
# %%
