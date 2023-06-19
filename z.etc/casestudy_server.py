# pyAPEP package import
import pyapep.isofit as isofit
import pyapep.simsep as simsep

# Data treatment package import
import numpy as np
import pandas as pd

# Data visualization package import
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager

# Data import
Data = pd.read_csv('Casestudy2_Zeolite13X.csv')

P_CO2 = Data['Pressure_CO2 (bar)'].dropna().values
q_CO2 = Data['Uptake_CO2 (mol/kg)'].dropna().values

P_CH4 = Data['Pressure_CH4 (bar)'].dropna().values
q_CH4 = Data['Uptake_CH4 (mol/kg)'].dropna().values

CO2_iso, CO2_p, CO2_name, CO2_err = isofit.best_isomodel(P_CO2, q_CO2)
CH4_iso, CH4_p, CO2_p, CH4_err = isofit.best_isomodel(P_CH4, q_CH4)
CO2_iso_ = lambda P,T: CO2_iso(P)
CH4_iso_ = lambda P,T: CH4_iso(P)


def MixIso(P, T):
    q1 = CO2_iso(P[0])
    q2 = CH4_iso(P[1])
    return q1, q2

import random

def simsep_cs(i):
    L = np.round(random.uniform(1, 10), 2)
    N = int((L/0.05)//10*10) +1
    R = np.round(random.uniform(0.1,0.5), 2)
    A_cros = np.pi*R**2
    
    CR1 = simsep.column(L, A_cros, n_component=2, N_node = N)
    ### Sorbent prop
    voidfrac = 0.37      # (m^3/m^3)
    D_particle = 12e-4   # (m)
    rho = 1324           # (kg/m^3)
    CR1.adsorbent_info(MixIso, epsi=voidfrac, D_particle=D_particle, rho_s=rho)

    ### Gas prop
    Mmol = [0.044, 0.016] # kg/mol
    mu_visco= [16.13E-6, 11.86E-6]   # (Pa sec) 
    CR1.gas_prop_info(Mmol, mu_visco)

    ### Transfer prop
    k_MTC  = [1E-2, 1E-2]     # m/sec==============> 값필요

    a_surf = 1 #Volumatric specific surface area (m2/m3)==============> 값필요 
    D_disp = [1E-2, 1E-2]     # m^2/sec ==============> 값필요 
    CR1.mass_trans_info(k_MTC, a_surf, D_disp)

    dH_ads = [31.164e3,20.856e3]   # J/mol
    Cp_s = 900
    Cp_g = [38.236, 35.8]  # J/mol/K

    h_heat = 100            # J/m2/K/s==============> 값필요 
    CR1.thermal_info(dH_ads, Cp_s, Cp_g, h_heat)
    
    ### Operating conditions
    P_inlet = random.randint(5, 20)
    P_outlet = P_inlet-1
    T_feed = 323
    y_feed = [0.33,0.67]

    Cv_inlet = 0.2E-1             # inlet valve constant (m/sec/bar)
    Cv_outlet= 2.0E-1           # outlet valve constant (m/sec/bar)

    Q_feed = 0.05*A_cros  # volumetric flowrate (m^3/sec)

    CR1.boundaryC_info(P_outlet, P_inlet, T_feed, y_feed,
                    Cv_inlet, Cv_outlet,
                    Q_inlet = Q_feed,
                    assigned_v_option = True,
                    foward_flow_direction = True)
    
    ### Initial conditions
    P_init = (P_inlet-0.5)*np.ones(N)    # (bar)
    y_init = [0.2*np.ones(N), 0.8*np.ones(N)] # (mol/mol)
    T_init = T_feed*np.ones(N)
    q_init = MixIso(P_init*np.array(y_init), T_init)

    CR1.initialC_info(P_init, T_init, T_init, y_init, q_init)

    y_res, z_res, t_res = CR1.run_mamoen(700,n_sec = 10, 
                                    CPUtime_print = True)
    
    bt = CR1.breakthrough(draw_graph = True)
    
    t_sim = np.arange(100, 700, 100)
    c1_frac = bt[0](t_sim)
    
    cs_res = [L, R, P_inlet, c1_frac]
    with open(f'results/{i}.pickle', wb) as fw:
        pickle.dump(cs_res, fw)
    
idx = np.arange(3)
with Pool(processes = cpu_count()) as p:
    p.map(simsep_cs, idx)

print(res_list)