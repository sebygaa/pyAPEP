"""
====================================================
Real PSA simulation module (:py:mod:`pyAPEP.simsep`)
====================================================

"""

import numpy as np
#from numpy.lib.function_base import _parse_input_dimensions
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
 
# %% Global varaiebls
R_gas = 8.3145      # 8.3145 J/mol/K
 
# %% column (bed) class
def Ergun(C_array,T_array, M_molar, mu_vis, D_particle,epsi_void,d,dd,d_fo, N):
    rho_g = np.zeros(N)
    for c, m in zip(C_array, M_molar):
        rho_g = rho_g + m*c
    P = np.zeros(N)
    for cc in C_array: 
        P = P + cc*R_gas*T_array
    dP = d@P
    
    Vs_Vg = (1-epsi_void)/epsi_void
    A =1.75*rho_g/D_particle*Vs_Vg
    B = 150*mu_vis/D_particle**2*Vs_Vg**2
    C = dP
 
    ind_posi = B**2-4*A*C >= 0
    ind_nega = ind_posi == False
    v_pos = (-B[ind_posi]+ np.sqrt(B[ind_posi]**2-4*A[ind_posi]*C[ind_posi]))/(2*A[ind_posi])
    v_neg = (B[ind_nega] - np.sqrt(B[ind_nega]**2+4*A[ind_nega]*C[ind_nega]))/(2*A[ind_nega])
    
    v_return = np.zeros(N)
    v_return[ind_posi] = v_pos
    v_return[ind_nega] = v_neg
    
    dv_return = d_fo@v_return
    return v_return, dv_return

def Ergun_test(dP,M_molar, mu_vis, D_particle,epsi_void):
    rho_g = 40*M_molar
    Vs_Vg = (1-epsi_void)/epsi_void
    A =1.75*rho_g/D_particle*Vs_Vg*np.ones_like(dP)
    B = 150*mu_vis/D_particle**2*Vs_Vg**2*np.ones_like(dP)
    C = dP
 
    ind_posi = B**2-4*A*C >= 0
    ind_nega = ind_posi == False
    v_pos = (-B[ind_posi]+ np.sqrt(B[ind_posi]**2-4*A[ind_posi]*C[ind_posi]))/(2*A[ind_posi])
    v_neg = (B[ind_nega] - np.sqrt(B[ind_nega]**2+4*A[ind_nega]*C[ind_nega]))/(2*A[ind_nega])
    
    v_return = np.zeros_like(dP)
    v_return[ind_posi] = v_pos
    v_return[ind_nega] = v_neg
    
    return v_return


def change_node_fn(z_raw, y_raw, N_new):
    if isinstance(y_raw,list):
        fn_list = []
        y_return = []
        z_new = np.linspace(z_raw[0], z_raw[-1],N_new)
        for yr in y_raw:
            fn_tmp = interp1d(z_raw,yr)
            y_new_tmp = fn_tmp(z_new)
            y_return.append(y_new_tmp)
    elif len(y_raw.shape) == 1:
        yy = y_raw
        fn_tmp = interp1d(z_raw, yy, kind = 'cubic')
        z_new = np.linspace(z_raw[0], z_raw[-1],N_new)
        y_return = fn_tmp(z_new)
    elif len(y_raw.shape) == 2:
        yy = y_raw[-1,:]
        fn_tmp = interp1d(z_raw, yy, kind = 'cubic')
        z_new = np.linspace(z_raw[0], z_raw[-1],N_new)
        y_return = fn_tmp(z_new)
    else:
        print('Input should be 1d or 2d array.')
        return None
    return y_return

# %% Column class
class column:
    """
    Instantiation. A `Column` class is for simulating packed bed column or pressure swing adsroption process.
    :param L: Length of column :math:`(m)`
    :param A_cross: Cross-sectional area of column :math:`(m^2)`
    :param n_component: Number of components 
    :param N_node: Number of nodes
    :param E_balance: Energy balance inclusion (default = True)  
    """    
    def __init__(self, L, A_cross, n_component, 
                 N_node = 11, E_balance = True):

                       
        self._L = L
        self._A = A_cross
        self._n_comp = n_component
        self._N = N_node
        self._z = np.linspace(0,L,N_node)
        self._required = {'Design':True,
        'adsorbent_info':False,
        'gas_prop_info': False,
        'mass_trans_info': False,}
        if E_balance:
            self._required['thermal_info'] = False
        self._required['boundaryC_info'] = False
        self._required['initialC_info'] = False
        h = L/(N_node-1)
        h_arr = h*np.ones(N_node)
        self._h = h
 
        # FDM backward, 1st deriv
        d0 = np.diag(1/h_arr, k = 0)
        d1 = np.diag(-1/h_arr[1:], k = -1)
        d = d0 + d1
        d[0,:] = 0
        self._d = d
        
        # FDM foward, 1st deriv
        d0_fo = np.diag(-1/h_arr, k = 0)
        d1_fo = np.diag(1/h_arr[1:], k = 1)
        d_fo = d0_fo + d1_fo
        self._d_fo  = d_fo
 
        # FDM centered, 2nd deriv
        dd0 = np.diag(1/h_arr[1:]**2, k = -1)
        dd1 = np.diag(-2/h_arr**2, k = 0)
        dd2 = np.diag(1/h_arr[1:]**2, k = 1)
        dd = dd0 + dd1 + dd2
        dd[0,:]  = 0
        dd[-1,:] = 0
        self._dd = dd
 
    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n'
        return str_return
 
 ### Before running the simulations ###

    def adsorbent_info(self, iso_fn, epsi = 0.3, D_particle = 0.01, rho_s = 1000,P_test_range=[0,10], T_test_range = [273,373]):
        """
        Adsorbent information
        
        :param iso_fn: Isothem function (Type: n_comp.N array, N array [pressure, temperature])
        :param epsi: Void fraction
        :param D_particle: Particle diameter :math:`(m)`
        :param rho_s: Solid density :math:`(kg/m^3)`
        :param P_test_range: Range of pressure for test :math:`(bar)`
        :param T_test_range: Range of temperature for test :math:`(K)`
        
        """
        T_test = np.linspace(T_test_range[0], T_test_range[1],self._N)
        p_test = []
        for ii in range(self._n_comp):
            p_tmp = P_test_range[0] + np.random.random(self._N)*(P_test_range[1] - P_test_range[0])
            p_test.append(p_tmp)        
        try:      
            iso_test = iso_fn(p_test, T_test)
            if len(iso_test) != self._n_comp:
                print('Output should be a list/narray including {} narray!'.format(self._n_comp))
            else:
                self._iso = iso_fn
                self._rho_s = rho_s
                self._epsi = epsi
                self._D_p =D_particle
                self._required['adsorbent_info'] = True
        except:
            print('You have problem in iso_fn')
            print('Input should be ( [p1_array,p2_array, ...] and T_array )')
            print('Output should be a list/narray including {} narray!'.format(self._n_comp))
        
        
    def gas_prop_info(self, Mass_molar, mu_viscosity):
        """
        Gas property information
        
        :param Mass_molar: Molar mass :math:`(mol/kg)`
        :param mu_viscosity: Viscosity :math:`(Pa \cdot s)`
        """
        stack_true = 0
        if len(Mass_molar) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if len(mu_viscosity) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if stack_true == 2:
            self._M_m = Mass_molar
            self._mu = mu_viscosity
            self._required['gas_prop_info'] = True
    def mass_trans_info(self, k_mass_transfer, a_specific_surf, D_dispersion = 1E-8):
        """
        Mass transfer
        
        :param k_mass_transfer: mass transfer coefficient :math:`(s^(-1))`
        :param a_specific_surf: specific surface area :math:`(m^2/m^3)`
        :param D_dispersion: dispersion coefficient :math:`(m^2/s)`
        """
        stack_true = 0
        if np.isscalar(D_dispersion):
            D_dispersion = D_dispersion*np.ones(self._n_comp)
        if len(k_mass_transfer) == self._n_comp:
            if np.isscalar(k_mass_transfer[0]):
                order = 1
                self._order_MTC = 1
            else:
                order = 2
                self._order_MTC = 2
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if len(D_dispersion) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if stack_true == 2:
            self._k_mtc = k_mass_transfer
            self._a_surf = a_specific_surf
            self._D_disp = D_dispersion
            self._required['mass_trans_info'] = True

    
    def thermal_info(self, dH_adsorption,
                     Cp_solid, Cp_gas, h_heat_transfer,
                     k_conduct = 0.0001, h_heat_ambient = 0.0, T_ambient = 298.15):
        """
        Thermal information
        
        :param dH_adsorption: Heat of adsorption :math:`(J/mol)`
        :param Cp_solid: Solid heat capacity :math:`(J/kg \cdot K)`
        :param Cp_gas: Gas heat capacity :math:`(J/mol \cdot K)` 
        :param h_heat_transfer: Heat transfer coefficient between solid and gas :math:`(J/m^2 \cdot K \cdot s)`  
        :param k_conduct: Conductivity of solid phase in axial direction :math:`(W/m \cdot K)`    
        :param h_heat_ambient: Heat transfer coefficient between ambient air and outer surface of the column :math:`(J/m^2 \cdot K \cdot s)`    
        :param T_ambient: Abient temperature :math:`(K)`
        """
        stack_true = 0
        n_comp = self._n_comp
        if len(dH_adsorption) != n_comp:
            print('dH_adsorption should be ({0:d},) list/narray.'.format(n_comp))
        else:
            stack_true = stack_true + 1
        if len(Cp_gas) != n_comp:
            print('Cp_gas should be ({0:d},) list/narray.'.format(n_comp))            
        else:
            stack_true = stack_true + 1
        if np.isscalar(h_heat_transfer):
            stack_true = stack_true + 1
        else:
            print('h_heat_transfer should be scalar.')
        if stack_true == 3:
            self._dH = dH_adsorption
            self._Cp_s = Cp_solid
            self._Cp_g = Cp_gas
            self._h_heat = h_heat_transfer
            self._k_cond = k_conduct
            self._h_ambi = h_heat_ambient
            self._T_ambi = T_ambient
            self._required['thermal_info'] = True
 
    def boundaryC_info(self, P_outlet,
    P_inlet,T_inlet,y_inlet, Cv_in=1E-1,Cv_out=1E-3,
      Q_inlet=None, assigned_v_option = True, 
      foward_flow_direction =  True):
        """
        Boundary condition information
        
        :param P_outlet: Outlet pressure :math:`(bar)` [scalar]
        :param P_inlet: Inlet pressure :math:`(bar)` [scalar]
        :param T_inlet: Inlet temperature :math:`(K)` [scalar] 
        :param y_inlet: Inlet composition :math:`(mol/mol)` [n_comp array]
        :param Cv_in: Valve constant of inlet side :math:`(m^3/bar \cdot s)` [scalar]  
        :param Cv_out: Valve constant of outlet side :math:`(mol/bar \cdot s)` [scalar]
        :param Q_inlet: Volumetric flow rate :math:`(m^3/s)`
        :param assigned_v_option: Assign velocity or not [Boolean]
        :param foward_flow_direction: Flow direction, if this is 'True' then the flow direction is foward.
        """
        self._Q_varying = False
        self._required['Flow direction'] = 'Foward'
        if foward_flow_direction == False:
            A_flip = np.zeros([self._N,self._N])
            for ii in range(self._N):
                A_flip[ii, -1-ii] = 1
            self._required['Flow direction'] = 'Backward'
            self._A_flip = A_flip
        if Q_inlet == None:
            assigned_v_option = False
        elif np.isscalar(Q_inlet) == False:
            assigned_v_option = True
            t = Q_inlet[0]
            Q = Q_inlet[1]
            f_Q_in = interp1d(t,Q)
            self._fn_Q = f_Q_in
            self._Q_varying = True
        try:
            if len(y_inlet) == self._n_comp:
                self._P_out = P_outlet
                self._P_in = P_inlet
                self._T_in = T_inlet
                self._y_in = y_inlet
                self._Q_in = Q_inlet
                self._Cv_in = Cv_in
                self._Cv_out = Cv_out
                self._const_v = assigned_v_option
                self._required['boundaryC_info'] = True
                if assigned_v_option:
                    self._required['Assigned velocity option'] = True
                else:
                    self._required['Assigned velocity option'] = False  
            else:
                print('The inlet composition should be a list/narray with shape (2, ).')
        except:
            print('The inlet composition should be a list/narray with shape (2, ).')    
 
    def initialC_info(self,P_initial, Tg_initial,Ts_initial, y_initial,q_initial):
        """
        Initial condition
        
        :param P_initial: Initial pressure :math:`(bar)` [N array]
        :param Tg_initial: Initial gas temperature :math:`(K)` [N array]
        :param Ts_initial: Initial solid temperature :math:`(K)` [N array]
        :param y_initial: Gas phase mol fraction :math:`(mol/mol)` [n_comp N array]
        :param q_initial: Solid phase uptake :math:`(mol/kg)` [n_comp N array]
        """
        stack_true = 0
        if len(P_initial) != self._N:
            print('P_initial should be of shape ({},)'.format(self._N))
        else:
            stack_true = stack_true + 1
        if len(y_initial) != self._n_comp or len(y_initial[0]) != self._N:
            print('y_initial should be a list including {0} ({1},) array'.format(self._n_comp, self._N))
        else:
            stack_true = stack_true + 1
        if len(q_initial) != self._n_comp or len(q_initial[0]) != self._N:
            print('q_initial should be a list/array including {0} ({1},) array'.format(self._n_comp, self._N))
        else:
            stack_true = stack_true + 1
        if stack_true == 3:
            self._P_init = P_initial
            self._Tg_init = Tg_initial
            self._Ts_init = Ts_initial
            self._y_init = y_initial
            self._q_init = q_initial
            self._required['initialC_info'] = True               

#########################
##### RUN FUNCTIONS #####
#########################

## Run mass & momentum balance equations
    def run_mamo(self, t_max, n_sec = 5, CPUtime_print = False):
        """
        Run mass & momentum balance equations
        
        :param t_max: Maximum time domain value 
        :param n_sec: Number of time node per second
        :param CPUtime_print: Print CPU time
     
        """
        tic = time.time()/60
        t_max_int = np.int32(np.floor(t_max), )
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        n_comp = self._n_comp
        C_sta = []
        for ii in range(n_comp):
            C_sta.append(self._y_in[ii]*self._P_in/R_gas/self._T_in*1E5)
        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        #print(C1_sta)

        # other parmeters
        epsi = self._epsi
        rho_s = self._rho_s

        N = self._N
        def massmomebal(y,t):
            C = []
            q = []
            for ii in range(n_comp):
                C.append(y[ii*N:(ii+1)*N])
                q.append(y[n_comp*N + ii*N : n_comp*N + (ii+1)*N])
 
            # Derivatives
            dC = []
            ddC = []
            C_ov = np.zeros(N)
            P_ov = np.zeros(N)
            P_part = []
            Mu = np.zeros(N)
            T = self._Tg_init
            for ii in range(n_comp):
                dC.append(self._d@C[ii])
                ddC.append(self._dd@C[ii])
                P_part.append(C[ii]*R_gas*T/1E5) # in bar
                C_ov = C_ov + C[ii]
                P_ov = P_ov + C[ii]*R_gas*T
                Mu = C[ii]*self._mu[ii]
            Mu = Mu/C_ov

            # Ergun equation
            v,dv = Ergun(C,T,self._M_m,Mu,self._D_p,epsi,
                         self._d,self._dd,self._d_fo, self._N)
            
            # Solid phase
            qsta = self._iso(P_part, T) # partial pressure in bar
            dqdt = []
            if self._order_MTC == 1:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii]*(qsta[ii] - q[ii])*self._a_surf
                    dqdt.append(dqdt_tmp)
            elif self._order_MTC == 2:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii][0]*(qsta[ii] - q[ii])*self._a_surf + self._k_mtc[ii][1]*(qsta[ii] - q[ii])**2*self._a_surf
                    dqdt.append(dqdt_tmp)
            # Valve equations (v_in and v_out)
            #P_in = (C1_sta + C2_sta)*R_gas*T_gas
            if self._const_v:
                v_in = self._Q_in/epsi/self._A
            else:
                v_in = max(self._Cv_in*(self._P_in - P_ov[0]/1E5), 0 )  # pressure in bar           
            v_out = max(self._Cv_out*(P_ov[-1]/1E5 - self._P_out), 0 )  # pressure in bar
            
            D_dis = self._D_disp
            h = self._h
            # Gas phase
            dCdt = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] -C[ii]*dv + D_dis[ii]*ddC[ii] - (1-epsi)/epsi*rho_s*dqdt[ii]
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v[-1]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = v_in*(C_sta[ii]-C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v_out+v[-1])/2*(C[ii][-2]- C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt.append(dCdt_tmp)
 
            dydt_tmp = dCdt+dqdt
            dydt = np.concatenate(dydt_tmp)
            return dydt
        
        C_init = []
        q_init = []
        for ii in range(n_comp):
            C_tmp = self._y_init[ii]*self._P_init*1E5/R_gas/self._Tg_init
            C_init.append(C_tmp)
            q_tmp = self._q_init[ii]
            q_init.append(q_tmp)
 
        tic = time.time()/60
        if self._required['Flow direction'] == 'Backward':
            for ii in range(n_comp):
                C_init[ii] = self._A_flip@C_init[ii]
                q_init[ii] = self._A_flip@q_init[ii]
        
        y0_tmp = C_init + q_init
        y0 = np.concatenate(y0_tmp)
        
        #RUN
        y_result = odeint(massmomebal,y0,t_dom,)
        
        if self._required['Flow direction'] == 'Backward':
            y_tmp = []
            for ii in range(n_comp*2):
                mat_tmp = y_result[:, ii*N : (ii+1)*N]
                y_tmp.append(mat_tmp@self._A_flip)
            y_flip = np.concatenate(y_tmp, axis = 1)
            y_result = y_flip
        self._y = y_result
        self._t = t_dom
        toc = time.time()/60 - tic
        self._CPU_min = toc
        self._Tg_res = np.ones([len(self._t), 1])@np.reshape(self._Tg_init,[1,-1])

        if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
        return y_result, self._z, t_dom

## Run mass & momentum & energy balance equations

    def run_mamoen_alt(self, t_max, n_sec = 5, CPUtime_print = False):
        """
        Run mass & momentum balance alternative
        
        :param t_max: Maximum time domain value 
        :param n_sec: Number of time node per second
        :param CPUtime_print: Print CPU time
     
        """
        t_max_int = np.int32(np.floor(t_max))
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        n_comp = self._n_comp
        
        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        N = self._N
        h = self._h
        epsi = self._epsi
        a_surf = self._a_surf

        # Mass
        D_dis = self._D_disp
        k_mass = self._k_mtc

        # Heat
        dH = self._dH
        Cpg = self._Cp_g
        Cps = self._Cp_s

        h_heat = self._h_heat
        h_ambi = self._h_ambi
        T_ambi = self._T_ambi

        # other parmeters
        rho_s = self._rho_s
        D_col = np.sqrt(self._A/np.pi)*2

        C_sta = []
        Cov_Cpg_in = 0
        for ii in range(n_comp):
            C_sta.append(self._y_in[ii]*self._P_in/R_gas/self._T_in*1E5)
            Cov_Cpg_in = Cov_Cpg_in + Cpg[ii]*C_sta[ii]
        
        def massmomeenerbal_alt(y,t):
            C = []
            q = []
            for ii in range(n_comp):
                C.append(y[ii*N:(ii+1)*N])
                q.append(y[n_comp*N + ii*N : n_comp*N + (ii+1)*N])
            Tg = y[2*n_comp*N : 2*n_comp*N + N ]
            Ts = y[2*n_comp*N + N : 2*n_comp*N + 2*N ]

 
            # Derivatives
            dC = []
            ddC = []
            C_ov = np.zeros(N)
            P_ov = np.zeros(N)
            P_part = []
            Mu = np.zeros(N)
            #T = self._Tg_init
            # Temperature gradient:
            dTg = self._d@Tg
            ddTs = self._dd@Ts

            # Concentration gradient
            # Pressure (overall&partial)
            # Viscosity
            for ii in range(n_comp):
                dC.append(self._d@C[ii])
                ddC.append(self._dd@C[ii])
                P_part.append(C[ii]*R_gas*Tg/1E5) # in bar
                C_ov = C_ov + C[ii]
                P_ov = P_ov + C[ii]*R_gas*Tg
                Mu = C[ii]*self._mu[ii]
            Mu = Mu/C_ov

            # Ergun equation
            v,dv = Ergun(C,Tg,self._M_m,Mu,self._D_p,epsi,
                         self._d,self._dd,self._d_fo, self._N)
            
            # Solid phase concentration
            qsta = self._iso(P_part, Tg) # partial pressure in bar
            dqdt = []
            if self._order_MTC == 1:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii]*(qsta[ii] - q[ii])*self._a_surf
                    dqdt.append(dqdt_tmp)
            elif self._order_MTC == 2:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii][0]*(qsta[ii] - q[ii])*self._a_surf + self._k_mtc[ii][1]*(qsta[ii] - q[ii])**2*self._a_surf
                    dqdt.append(dqdt_tmp)
            # Valve equations (v_in and v_out)
            #P_in = (C1_sta + C2_sta)*R_gas*T_gas
            if self._const_v:
                v_in = self._Q_in/epsi/self._A
            else:
                v_in = max(self._Cv_in*(self._P_in - P_ov[0]/1E5), 0 )  # pressure in bar           
            v_out = max(self._Cv_out*(P_ov[-1]/1E5 - self._P_out), 0 )  # pressure in bar
            
            # Gas phase concentration
            dCdt = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] -C[ii]*dv + D_dis[ii]*ddC[ii] - (1-epsi)/epsi*rho_s*dqdt[ii]
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]    
                dCdt_tmp[0] = +v_in*(C_sta[ii] - C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v_out+v[-1])/2*(C[ii][-2]- C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt.append(dCdt_tmp)
            # Temperature (gas)
            Cov_Cpg = np.zeros(N) # Heat capacity (overall) J/K/m^3
            for ii in range(n_comp):
                Cov_Cpg = Cov_Cpg + Cpg[ii]*C[ii]
            dTgdt = -v*dTg + h_heat*a_surf/epsi*(Ts - Tg)/Cov_Cpg
            for ii in range(n_comp):
                dTgdt = dTgdt - Cpg[ii]*Tg*D_dis[ii]*ddC[ii]/Cov_Cpg
                dTgdt = dTgdt + Tg*self._rho_s*(1-epsi)/epsi*Cpg[ii]*dqdt[ii]/Cov_Cpg
                dTgdt = dTgdt + h_ambi*4/epsi/D_col*(T_ambi - Tg)/Cov_Cpg

            dTgdt[0] = h_heat*a_surf/epsi*(Ts[0] - Tg[0])/Cov_Cpg[0]
            dTgdt[0] = dTgdt[0] + v_in*(self._T_in - Tg[0])/h
            dTgdt[-1] = h_heat*a_surf/epsi*(Ts[-1] - Tg[-1])/Cov_Cpg[-1]
            #dTgdt[-1] = dTgdt[-1] + (v[-1]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            dTgdt[-1] = dTgdt[-1] + (v[-1]+v_out)/2*(Tg[-2]-Tg[-1])/h
            for ii in range(n_comp):
                dTgdt[0] = dTgdt[0] - Tg[0]*Cpg[ii]*dCdt[ii][0]/Cov_Cpg[0]
                dTgdt[-1] = dTgdt[-1] - Tg[-1]*Cpg[ii]*dCdt[ii][-1]/Cov_Cpg[-1]
            dTsdt = (self._k_cond*ddTs+ h_heat*a_surf/(1-epsi)*(Tg-Ts))/self._rho_s/Cps
            for ii in range(n_comp):
                dTsdt = dTsdt + abs(dH[ii])*dqdt[ii]/Cps
            
            dydt_tmp = dCdt+dqdt+[dTgdt] + [dTsdt]
            dydt = np.concatenate(dydt_tmp)
            return dydt
        
        C_init = []
        q_init = []
        for ii in range(n_comp):
            C_tmp = self._y_init[ii]*self._P_init*1E5/R_gas/self._Tg_init
            C_init.append(C_tmp)
            q_tmp = self._q_init[ii]
            q_init.append(q_tmp)
 
        tic = time.time()/60
        if self._required['Flow direction'] == 'Backward':
            for ii in range(n_comp):
                C_init[ii] = self._A_flip@C_init[ii]
                q_init[ii] = self._A_flip@q_init[ii]
            y0_tmp = C_init + q_init + [self._A_flip@self._Tg_init] + [self._A_flip@self._Ts_init]
        else:
            y0_tmp = C_init + q_init + [self._Tg_init] + [self._Ts_init]
        y0 = np.concatenate(y0_tmp)
        
        #RUN
        y_result = odeint(massmomeenerbal_alt,y0,t_dom,)
        
        if self._required['Flow direction'] == 'Backward':
            y_tmp = []
            for ii in range(n_comp*2 + 2):
                mat_tmp = y_result[:, ii*N : (ii+1)*N]
                y_tmp.append(mat_tmp@self._A_flip)
            y_flip = np.concatenate(y_tmp, axis = 1)
            y_result = y_flip
        self._y = y_result
        self._t = t_dom
        toc = time.time()/60 - tic
        self._CPU_min = toc
        self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
        if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
        return y_result, self._z, t_dom



    def run_mamoen(self, t_max, n_sec = 5, CPUtime_print = False):
        """
        Run mass & momentum & energy balance 
        
        :param t_max: Maximum time domain value 
        :param n_sec: Number of time node per second
        :param CPUtime_print: Print CPU time
     
        """
      
        t_max_int = np.int32(np.floor(t_max))
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        n_comp = self._n_comp
        
        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        N = self._N
        h = self._h
        epsi = self._epsi
        a_surf = self._a_surf

        # Mass
        D_dis = self._D_disp
        k_mass = self._k_mtc

        # Heat
        dH = self._dH
        Cpg = self._Cp_g
        Cps = self._Cp_s

        h_heat = self._h_heat
        h_ambi = self._h_ambi
        T_ambi = self._T_ambi

        # other parmeters
        rho_s = self._rho_s
        D_col = np.sqrt(self._A/np.pi)*2

        C_sta = []
        Cov_Cpg_in = 0
        for ii in range(n_comp):
            C_sta.append(self._y_in[ii]*self._P_in/R_gas/self._T_in*1E5)
            Cov_Cpg_in = Cov_Cpg_in + Cpg[ii]*C_sta[ii]
        
        def massmomeenerbal(y,t):
            C = []
            q = []
            for ii in range(n_comp):
                C.append(y[ii*N:(ii+1)*N])
                q.append(y[n_comp*N + ii*N : n_comp*N + (ii+1)*N])
            Tg = y[2*n_comp*N : 2*n_comp*N + N ]
            Ts = y[2*n_comp*N + N : 2*n_comp*N + 2*N ]

 
            # Derivatives
            dC = []
            ddC = []
            C_ov = np.zeros(N)
            P_ov = np.zeros(N)
            P_part = []
            Mu = np.zeros(N)
            #T = self._Tg_init
            # Temperature gradient:
            dTg = self._d@Tg
            ddTs = self._dd@Ts

            # Concentration gradient
            # Pressure (overall&partial)
            # Viscosity
            for ii in range(n_comp):
                dC.append(self._d@C[ii])
                ddC.append(self._dd@C[ii])
                P_part.append(C[ii]*R_gas*Tg/1E5) # in bar
                C_ov = C_ov + C[ii]
                P_ov = P_ov + C[ii]*R_gas*Tg
                Mu = C[ii]*self._mu[ii]
            Mu = Mu/C_ov

            # Ergun equation
            v,dv = Ergun(C,Tg,self._M_m,Mu,self._D_p,epsi,
                         self._d,self._dd,self._d_fo, self._N)
            
            # Solid phase concentration
            qsta = self._iso(P_part, Tg) # partial pressure in bar
            dqdt = []
            if self._order_MTC == 1:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii]*(qsta[ii] - q[ii])*self._a_surf
                    dqdt.append(dqdt_tmp)
            elif self._order_MTC == 2:
                for ii in range(n_comp):
                    dqdt_tmp = self._k_mtc[ii][0]*(qsta[ii] - q[ii])*self._a_surf + self._k_mtc[ii][1]*(qsta[ii] - q[ii])**2*self._a_surf
                    dqdt.append(dqdt_tmp)
            # Valve equations (v_in and v_out)
            #P_in = (C1_sta + C2_sta)*R_gas*T_gas
            if self._const_v:
                v_in = self._Q_in/epsi/self._A
            else:
                v_in = max(self._Cv_in*(self._P_in - P_ov[0]/1E5), 0 )  # pressure in bar           
            v_out = max(self._Cv_out*(P_ov[-1]/1E5 - self._P_out), 0 )  # pressure in bar
            
            # Gas phase concentration
            dCdt = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] -C[ii]*dv + D_dis[ii]*ddC[ii] - (1-epsi)/epsi*rho_s*dqdt[ii]
                dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]    
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-1]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt.append(dCdt_tmp)
            # Temperature (gas)
            Cov_Cpg = np.zeros(N) # Heat capacity (overall) J/K/m^3
            for ii in range(n_comp):
                Cov_Cpg = Cov_Cpg + Cpg[ii]*C[ii]
            dTgdt = -v*dTg + h_heat*a_surf/epsi*(Ts - Tg)/Cov_Cpg
            for ii in range(n_comp):
                dTgdt = dTgdt - Cpg[ii]*Tg*D_dis[ii]*ddC[ii]/Cov_Cpg
                dTgdt = dTgdt + Tg*self._rho_s*(1-epsi)/epsi*Cpg[ii]*dqdt[ii]/Cov_Cpg
                dTgdt = dTgdt + h_ambi*4/epsi/D_col*(T_ambi - Tg)/Cov_Cpg

            dTgdt[0] = h_heat*a_surf/epsi*(Ts[0] - Tg[0])/Cov_Cpg[0]
            dTgdt[0] = dTgdt[0] + (v_in*self._T_in*Cov_Cpg_in/Cov_Cpg[0] - v[1]*Tg[0])/h
            dTgdt[-1] = h_heat*a_surf/epsi*(Ts[-1] - Tg[-1])/Cov_Cpg[-1]
            #dTgdt[-1] = dTgdt[-1] + (v[-1]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            dTgdt[-1] = dTgdt[-1] + (v[-1]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            for ii in range(n_comp):
                dTgdt[0] = dTgdt[0] - Tg[0]*Cpg[ii]*dCdt[ii][0]/Cov_Cpg[0]
                dTgdt[-1] = dTgdt[-1] - Tg[-1]*Cpg[ii]*dCdt[ii][-1]/Cov_Cpg[-1]
            dTsdt = (self._k_cond*ddTs+ h_heat*a_surf/(1-epsi)*(Tg-Ts))/self._rho_s/Cps
            for ii in range(n_comp):
                dTsdt = dTsdt + abs(dH[ii])*dqdt[ii]/Cps
            
            dydt_tmp = dCdt+dqdt+[dTgdt] + [dTsdt]
            dydt = np.concatenate(dydt_tmp)
            return dydt
        
        C_init = []
        q_init = []
        for ii in range(n_comp):
            C_tmp = self._y_init[ii]*self._P_init*1E5/R_gas/self._Tg_init
            C_init.append(C_tmp)
            q_tmp = self._q_init[ii]
            q_init.append(q_tmp)
 
        tic = time.time()/60
        if self._required['Flow direction'] == 'Backward':
            for ii in range(n_comp):
                C_init[ii] = self._A_flip@C_init[ii]
                q_init[ii] = self._A_flip@q_init[ii]
            y0_tmp = C_init + q_init + [self._A_flip@self._Tg_init] + [self._A_flip@self._Ts_init]
        else:
            y0_tmp = C_init + q_init + [self._Tg_init] + [self._Ts_init]
        y0 = np.concatenate(y0_tmp)
        
        #RUN
        y_result = odeint(massmomeenerbal,y0,t_dom,)
        C_sum = 0
        for ii in range(n_comp):
            C_sum = C_sum + y_result[-1,ii*N+2]
        if C_sum  < 0.1:
            y_result, _,_  = self.run_mamoen_alt(t_max,n_sec)
            toc = time.time()/60 - tic
            self._CPU_min = toc
            self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))
            return y_result
        
        if self._required['Flow direction'] == 'Backward':
            y_tmp = []
            for ii in range(n_comp*2 + 2):
                mat_tmp = y_result[:, ii*N : (ii+1)*N]
                y_tmp.append(mat_tmp@self._A_flip)
            y_flip = np.concatenate(y_tmp, axis = 1)
            y_result = y_flip
        self._y = y_result
        self._t = t_dom
        toc = time.time()/60 - tic
        self._CPU_min = toc
        self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
        if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
        return y_result, self._z, t_dom


## Functions for after-run processing
    def next_init(self, change_init = True):
        """
        Next initalization
        
        :param change_init: Replace inital condition with previous result at final time [boolean]
        :return: Previous result at final time (if this is 'True', initial conditions are repalced automatically)
        
        """
        N = self._N
        y_end = self._y[-1,:]
        C = []
        q = []
        y = []
        C_ov = np.zeros(N)
        cc = 0
        P_ov = np.zeros(N)
        for ii in range(self._n_comp):
            C_tmp = y_end[cc*N:(cc+1)*N]
            C.append(C_tmp)
            C_ov = C_ov + C_tmp
            P_ov = P_ov + C_tmp*R_gas*self._Tg_res[-1,:]/1E5
            cc = cc+1
        for ii in range(self._n_comp):
            q_tmp = y_end[cc*N:(cc+1)*N]
            y_tmp = C[ii]/C_ov
            q.append(q_tmp)
            y.append(y_tmp)
            cc = cc + 1
        try:
            if self._required['thermal_info']:
                Tg_init = y_end[cc*N:(cc+1)*N]    
                Ts_init = y_end[(cc+1)*N:(cc+2)*N]    
            else:
                Tg_init = self._Tg_res[-1,:]    
                Ts_init = self._Tg_res[-1,:]    
        except:
            Tg_init = self._Tg_res[-1,:]
            Ts_init = self._Tg_res[-1,:]
        P_init = P_ov
        y_init = y
        q_init = q

        if change_init:
            self._P_init = P_init
            self._Tg_init= Tg_init
            self._Ts_init = Ts_init
            self._y_init = y_init
            self._q_init = q_init 
        return P_init, Tg_init, Ts_init , y_init, q_init
    
    def change_init_node(self, N_new):
        """
        Initial node change
        
        :param N_new: New number of nodes
        
        """
        if self._N == N_new:
            print('No change in # of node.')
            return
        else:
            z = self._z
            P_init = self._P_init
            Tg_init = self._Tg_init
            Ts_init = self._Ts_init
            y_init = self._y_init
            q_init = self._q_init
        P_new = change_node_fn(z, P_init, N_new)
        Tg_new = change_node_fn(z,Tg_init, N_new)
        Ts_new = change_node_fn(z,Ts_init, N_new)
        y_new = change_node_fn(z,y_init, N_new)
        q_new = change_node_fn(z,q_init, N_new)
        self._z = np.linspace(0, z[-1], N_new)

        self._P_init = P_new 
        self._Tg_init = Tg_new
        self._Ts_init = Ts_new
        self._y_init = y_new
        self._q_init = q_new

        self._N = N_new
        h_arr = z[-1]/(N_new-1)*np.ones(N_new)
        # FDM backward, 1st deriv
        d0 = np.diag(1/h_arr, k = 0)
        d1 = np.diag(-1/h_arr[1:], k = -1)
        d = d0 + d1
        d[0,:] = 0
        self._d = d
        
        # FDM foward, 1st deriv
        d0_fo = np.diag(-1/h_arr, k = 0)
        d1_fo = np.diag(1/h_arr[1:], k = 1)
        d_fo = d0_fo + d1_fo
        self._d_fo  = d_fo
 
        # FDM centered, 2nd deriv
        dd0 = np.diag(1/h_arr[1:]**2, k = -1)
        dd1 = np.diag(-2/h_arr**2, k = 0)
        dd2 = np.diag(1/h_arr[1:]**2, k = 1)
        dd = dd0 + dd1 + dd2
        dd[0,:]  = 0
        dd[-1,:] = 0
        self._dd = dd

    def Q_valve(self, draw_graph = False, y = None):
        """
        Q valve
        
        :param draw_graph: Show graph [boolean] 
        :param y: Simulation result of run_mamo and run_mamoen (if it is 'None', this value is from the simulation result automatically)
        
        :return: Volumetric flow rates at Z = 0 and L [time_node array]
        """
        N = self._N
        if self._required['Flow direction'] == 'Backward':
            Cv_0 = self._Cv_out
            Cv_L = self._Cv_in
        else:
            Cv_0 = self._Cv_in
            Cv_L = self._Cv_out
            
        if y == None:
            y = self._y
        P_0 = (y[:,0])*R_gas*self._Tg_res[:,0]/1E5
        P_L = (y[:,N-1])*R_gas*self._Tg_res[:,-1]/1E5
        for ii in range(1,self._n_comp):
            P_0 = P_0 + y[:,N*ii]*R_gas*self._Tg_res[:,0]/1E5
            P_L = P_L + y[:,N*ii+N-1]*R_gas*self._Tg_res[:,-1]/1E5
        
        if self._required['Flow direction'] == 'Backward':
            if self._required['Assigned velocity option']:
                v_L = self._Q_in*np.ones(len(self._t))/self._A/self._epsi
            else:
                v_L = Cv_L*(self._P_in - P_L)
            v_0 = Cv_0*(P_0 - self._P_out)
        else:
            if self._required['Assigned velocity option']:
                v_0 = self._Q_in*np.ones(len(self._t))/self._A/self._epsi
            else:
                v_0 = Cv_0*(self._P_in - P_0)
            v_L = Cv_L*(P_L - self._P_out)
        Q_0 = v_0*self._A * self._epsi
        Q_0[Q_0 < 0] = 0
        Q_L = v_L*self._A * self._epsi
        Q_L[Q_L < 0] = 0
 
        if draw_graph:
            plt.figure(figsize = [6.5,5], dpi = 90)
            plt.plot(self._t, Q_0,
            label = 'Q at z = 0', linewidth = 2)
            plt.plot(self._t, Q_L, 
            label = 'Q at z = L', linewidth = 2)
            plt.legend(fontsize = 14)
            plt.xlabel('time (sec)', fontsize = 15)
            plt.ylabel('volumetric flowrate (m$^{3}$/sec)', fontsize =15)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
        return Q_0, Q_L
    
    def breakthrough(self, draw_graph = False):
        """
        Breakthrough
       
     
        """
        N = self._N
        n_comp = self._n_comp
        C_end =[]
        C_ov = np.zeros(len(self._t))
        for ii in range(n_comp):
            Cend_tmp = self._y[:,(ii+1)*N-1]
            C_end.append(Cend_tmp)
            C_ov = C_ov + Cend_tmp
        y = []
        fn_y = [] 
        for ii in range(n_comp):
            y_tmp = C_end[ii]/C_ov
            y.append(y_tmp)
            fn_tmp = interp1d(self._t, y_tmp, kind = 'cubic' ) 
            fn_y.append(fn_tmp)
        if draw_graph:
            t_dom = np.linspace(self._t[0], self._t[-1],1001)
            y_again = []
            plt.figure(figsize = [8,5], dpi = 90)
            for ii in range(n_comp):
                y_ag_tmp = fn_y[ii](t_dom)
                y_again.append(y_ag_tmp)
                plt.plot(t_dom, y_ag_tmp,
                         label = 'Component{0:2d}'.format(ii+1),
                         linewidth = 2 )            
            plt.legend(fontsize = 14)
            plt.xlabel('time (sec)', fontsize = 15)
            plt.ylabel('mole fraction (mol/mol)', fontsize =15)
            plt.xticks(fontsize = 14)
            plt.yticks(fontsize = 14)
    
        return fn_y
 
    def Graph(self, every_n_sec, index, 
              loc = [1,1], yaxis_label = None, 
              file_name = None, 
              figsize = [7,5], dpi = 85, y = None,):
        """
        Making graph
        
        :param every_n_sec: Number of points in graph 
        :param index: Index determining which graph is to be displayed = 0 - n_comp-1 gas concentration :math:`(mol/m^3)` / n_comp - 2 n_comp - 1 solid phase uptake :math:`(mol/kg)` / 2 n_comp gas phase temperature :math:`(K)` / 2 n_comp + 1 solid phase temeprature 
        :param loc: Location of legend
        :param yaxis_label: ylabel of graph
        :param file_name: File name
        :param figsize: Figure size (default [7,5])
        :param dpi: Dot per inch (default 85)       
        :param y: Gas phase mol fraction :math:`(mol/mol)` [n_comp N array]
     
        """
        N = self._N
        one_sec = self._n_sec
        n_show = one_sec*every_n_sec
        if y == None:
            y = self._y
        lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']
        fig, ax = plt.subplots(figsize = figsize, dpi = 90)
        cc= 0
        lcolor = 'k'
        for j in range(0,len(self._y), n_show):
            if j <= 1:
                lcolor = 'r'
            elif j >= len(self._y)-n_show:
                lcolor = 'b'
            else:
                lcolor = 'k'
            ax.plot(self._z,self._y[j, index*N:(index+1)*N],
            color = lcolor, linestyle = lstyle[cc%len(lstyle)],
            label = 't = {}'.format(self._t[j]))
            cc = cc + 1
        fig.legend(fontsize = 14,bbox_to_anchor = loc)
        ax.set_xlabel('z-domain (m)', fontsize = 15)
        if yaxis_label == None:
            ylab = 'Variable index = {}'.format(index)
            ax.set_ylabel(ylab, fontsize = 15)
        else:
            ax.set_ylabel(yaxis_label, fontsize = 15)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.grid(linestyle = ':')
        if file_name != None:
            fig.savefig(file_name, bbox_inches='tight')
        
        return fig, ax
        
    def Graph_P(self, every_n_sec, loc = [1,1], 
                yaxis_label = 'Pressure (bar)',
                file_name = None, 
                figsize = [7,5], dpi = 85, y = None,):
        """
        Making graph of partial pressure
        
        :param every_n_sec: Number of points in graph 
        :param loc: Location of legend
        :param yaxis_label: ylabel of graph
        :param file_name: File name
        :param figsize: Figure size (default [7,5])
        :param dpi: Dot per inch (default 85)       
        :param y: Gas phase mol fraction :math:`(mol/mol)` [n_comp N array]
        """
        N = self._N
        one_sec = self._n_sec
        n_show = one_sec*every_n_sec
        if y == None:
            y = self._y
        lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':']
        P = np.zeros(N)
        for ii in range(self._n_comp):
            P = P + self._y[:,(ii)*N:(ii+1)*N]*R_gas*self._Tg_res/1E5
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        cc= 0
        for j in range(0,len(self._y), n_show):
            if j <= 1:
                lcolor = 'r'
            elif j >= len(self._y)-n_show:
                lcolor = 'b'
            else:
                lcolor = 'k'
            ax.plot(self._z,P[j, :],
            color = lcolor, linestyle = lstyle[cc%len(lstyle)],
            label = 't = {}'.format(self._t[j]))
            cc = cc + 1
        fig.legend(fontsize = 14, bbox_to_anchor = loc)
        ax.set_xlabel('z-domain (m)', fontsize = 15)
        ax.set_ylabel(yaxis_label, fontsize = 15)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.grid(linestyle = ':')
        if file_name != None:
            fig.savefig(file_name, bbox_inches='tight')
        return fig, ax    
    ## Copy the         
    def copy(self):
        """
        Copy
        """
        import copy
        self_new = copy.deepcopy(self)
        return self_new




def step_P_eq_alt1(column1, column2, t_max,
n_sec=5, Cv_btw=0.1, valve_select = [1,1], CPUtime_print = False):
    """
 
    """
    tic = time.time() / 60 # in minute
    P_sum1 = np.mean(column1._P_init)
    P_sum2 = np.mean(column2._P_init)
    P_mean = (P_sum1+P_sum2)/2
    T_mean = (np.mean(column1._Tg_init)+np.mean(column2._Tg_init))/2
    if P_sum1 > P_sum2:
        c1_tmp = column1.copy()
        c2_tmp = column2.copy()
        val_sel = np.array(valve_select)
        switch_later = False
    else:
        c1_tmp = column2.copy()
        c2_tmp = column1.copy()
        val_sel = np.array([valve_select[0], valve_select[1]])
        switch_later = True
    if val_sel[0] == 0:
        A_flip1 = np.zeros([c1_tmp._N,c1_tmp._N])
        for ii in range(c1_tmp._N):
            A_flip1[ii, -1-ii] = 1
        c1_tmp._P_init =c1_tmp._P_init@A_flip1
        c1_tmp._Tg_init=c1_tmp._Tg_init@A_flip1
        c1_tmp._Ts_init=c1_tmp._Ts_init@A_flip1
        c1_tmp._y_init=c1_tmp._y_init@A_flip1
        c1_tmp._q_init=c1_tmp._q_init@A_flip1
        flip1_later = True
    else:
        flip1_later = False
    if val_sel[1]:
        A_flip2 = np.zeros([c2_tmp._N,c2_tmp._N])
        for ii in range(c2_tmp._N):
            A_flip2[ii, -1-ii] = 1
        c2_tmp._P_init =c2_tmp._P_init@A_flip2
        c2_tmp._Tg_init=c2_tmp._Tg_init@A_flip2
        c2_tmp._Ts_init=c2_tmp._Ts_init@A_flip2
        c2_tmp._y_init=c2_tmp._y_init@A_flip2
        c2_tmp._q_init=c2_tmp._q_init@A_flip2
        flip2_later=True
    else:
        flip2_later = False
    t_max_int = np.int32(np.floor(t_max))
    c1_tmp._n_sec = n_sec
    c2_tmp._n_sec = n_sec
    n_t = t_max_int*n_sec+ 1
    n_comp = column1._n_comp
    
    # Target pressure
    C_sta1 = np.zeros(n_comp)
    for ii in range(n_comp):
        C_sta1[ii] = c1_tmp._y_init[ii][0]*P_mean/R_gas/T_mean*1E5
    print(C_sta1)

    t_dom = np.linspace(0,t_max_int, n_t)
    column1._n_sec = n_sec
    column2._n_sec = n_sec
    if t_max_int < t_max:
        t_dom = np.concatenate((t_dom, [t_max]))

    N1 = c1_tmp._N
    N2 = c2_tmp._N
    h1 = c1_tmp._h
    h2 = c2_tmp._h
    epsi1 = c1_tmp._epsi
    epsi2 = c2_tmp._epsi

    a_surf1 = c1_tmp._a_surf
    a_surf2 = c2_tmp._a_surf
    # Mass
    D_dis1 = c1_tmp._D_disp
    D_dis2 = c2_tmp._D_disp

    k_mass1 = c1_tmp._k_mtc
    k_mass2 = c2_tmp._k_mtc

    # Heat
    dH1 = c1_tmp._dH
    dH2 = c2_tmp._dH
    Cpg1 = c1_tmp._Cp_g
    Cpg2 = c2_tmp._Cp_g
    
    Cps1 = c1_tmp._Cp_s
    Cps2 = c2_tmp._Cp_s

    h_heat1 = c1_tmp._h_heat
    h_heat2 = c2_tmp._h_heat

    h_ambi1 = c1_tmp._h_ambi
    h_ambi2 = c2_tmp._h_ambi

    T_ambi1 = c1_tmp._T_ambi
    T_ambi2 = c2_tmp._T_ambi

    # other parmeters
    rho_s1 = c1_tmp._rho_s
    rho_s2 = c2_tmp._rho_s
    D_col1 = np.sqrt(c1_tmp._A/np.pi)*2
    D_col2 = np.sqrt(c2_tmp._A/np.pi)*2

    n_var_tot1 = c1_tmp._N * (c1_tmp._n_comp+1)*2
    n_var_tot2 = c2_tmp._N * (c2_tmp._n_comp+1)*2
    
    # Initial conditions
    y01 = []
    y02 = []
    for ii in range(n_comp):
        C1_tmp = c1_tmp._P_init/R_gas/c1_tmp._Tg_init*c1_tmp._y_init[ii]*1E5    # in (mol/m^3)
        C2_tmp = c2_tmp._P_init/R_gas/c2_tmp._Tg_init*c2_tmp._y_init[ii]*1E5    # in (mol/m^3)
        y01.append(C1_tmp)
        y02.append(C2_tmp)
    for ii in range(n_comp):
        y01.append(c1_tmp._q_init[ii])
        y02.append(c2_tmp._q_init[ii])
    y01.append(c1_tmp._Tg_init)
    y02.append(c2_tmp._Tg_init)
    y01.append(c1_tmp._Ts_init)
    y02.append(c2_tmp._Ts_init)
    y01 = np.concatenate(y01)
    y02 = np.concatenate(y02)
    y0_tot = np.concatenate([y01,y02])
    
    # ODE function (Gas only)
    def massmomeenbal_eq_gasonly(y,t):
        y1 = y[:N1*n_comp]
        y2 = y[N1*n_comp:]
        C1 = []
        C2 = []
        for ii in range(n_comp):
            C1.append(y1[ii*N1:(ii+1)*N1])
            C2.append(y2[ii*N2:(ii+1)*N2])
            
        # Derivatives
        dC1 = []
        dC2 = []
        ddC1 = []
        ddC2 = []
        C_ov1 = np.zeros(N1)
        C_ov2 = np.zeros(N2)
        P_ov1 = np.zeros(N1)
        P_ov2 = np.zeros(N2)
        P_part1 = []
        P_part2 = []
        Mu1 = np.zeros(N1)
        Mu2 = np.zeros(N2)
        #T = self._Tg_init
        # Temperature gradient:
        
        # Concentration gradient
        # Pressure (overall&partial)
        # Viscosity
        for ii in range(n_comp):
            dC1.append(c1_tmp._d@C1[ii])
            dC2.append(c2_tmp._d@C2[ii])
            ddC1.append(c1_tmp._dd@C1[ii])
            ddC2.append(c2_tmp._dd@C2[ii])
            P_part1.append(C1[ii]*R_gas*T_mean/1E5) # in bar
            P_part2.append(C2[ii]*R_gas*T_mean/1E5) # in bar
            C_ov1 = C_ov1 + C1[ii]
            C_ov2 = C_ov2 + C2[ii]
            P_ov1 = P_ov1 + C1[ii]*R_gas*T_mean
            P_ov2 = P_ov2 + C2[ii]*R_gas*T_mean
            Mu1 = C1[ii]*c1_tmp._mu[ii]
            Mu2 = C2[ii]*c2_tmp._mu[ii]
        Mu1 = Mu1/C_ov1
        Mu2 = Mu2/C_ov2

        # Ergun equation
        v1,dv1 = Ergun(C1,T_mean,c1_tmp._M_m,Mu1,c1_tmp._D_p,epsi1,
        c1_tmp._d,c1_tmp._dd,c1_tmp._d_fo, N1)
        v2,dv2 = Ergun(C2,T_mean,c2_tmp._M_m,Mu2,c2_tmp._D_p,epsi2,
        c2_tmp._d,c2_tmp._dd,c2_tmp._d_fo, N2)
        
        # Valve equations (v_in and v_out)
        v_in1 = 0
        v_out2 = 0
        
        v_out1 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar
        v_in2 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar           
        
        # Gas phase concentration
        dCdt1 = []
        dCdt2 = []
        for ii in range(n_comp):
            dCdt_tmp = -v1*dC1[ii] -C1[ii]*dv1 + D_dis1[ii]*ddC1[ii]
            #dCdt_tmp[0] = +(v_in1*0 - v1[1]*C1[ii][0])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]               ######BOUNDARY C########
            dCdt_tmp[0] = +Cv_btw*(C_sta1[ii] - C1[ii][0])               ######BOUNDARY C########
            dCdt_tmp[-1]= +(v1[-1]*C1[ii][-2]-v_out1*C1[ii][-1])/h1        ######BOUNDARY C######## (KEY PROBLEM)
            dCdt1.append(dCdt_tmp)
        inout_ratio_mass=epsi1*c1_tmp._A/epsi2/c2_tmp._A
        for ii in range(n_comp):
            dCdt_tmp = -v2*dC2[ii] -C2[ii]*dv2 + D_dis2[ii]*ddC2[ii]
            dCdt_tmp[0] = +(v_in2*inout_ratio_mass*C1[ii][-1] - v2[1]*C2[ii][0])/h2 ######BOUNDARY C########
            dCdt_tmp[-1]= +(v2[-1]*C2[ii][-2]- v_out2*C2[ii][-1])/h2                ######BOUNDARY C########
            dCdt2.append(dCdt_tmp)

        # Temperature (gas)
        Cov_Cpg1 = np.zeros(N1) # Heat capacity (overall) J/K/m^3
        Cov_Cpg2 = np.zeros(N2) # Heat capacity (overall) J/K/m^3
        for ii in range(n_comp):
            Cov_Cpg1 = Cov_Cpg1 + Cpg1[ii]*C1[ii]
            Cov_Cpg2 = Cov_Cpg2 + Cpg2[ii]*C2[ii]

        dydt_tmp1 = dCdt1
        dydt_tmp2 = dCdt2
        dydt1 = np.concatenate(dydt_tmp1)
        dydt2 = np.concatenate(dydt_tmp2)
        dydt = np.concatenate([dydt1,dydt2])
        return dydt

    # ODE function (Gas and Solid)
    def massmomeenbal_eq(y,t):
        y1 = y[:n_var_tot1]
        y2 = y[n_var_tot1:]
        C1 = []
        C2 = []
        q1 = []
        q2 = []
        for ii in range(n_comp):
            C1.append(y1[ii*N1:(ii+1)*N1])
            C2.append(y2[ii*N2:(ii+1)*N2])
            q1.append(y1[n_comp*N1 + ii*N1 : n_comp*N1 + (ii+1)*N1])
            q2.append(y2[n_comp*N2 + ii*N2 : n_comp*N2 + (ii+1)*N2])
        Tg1 =y1[2*n_comp*N1 : 2*n_comp*N1 + N1 ]
        Tg2 =y2[2*n_comp*N2 : 2*n_comp*N2 + N2 ]
        Ts1 =y1[2*n_comp*N1 + N1 : 2*n_comp*N1 + 2*N1 ]
        Ts2 =y2[2*n_comp*N2 + N2 : 2*n_comp*N2 + 2*N2 ]

        # Derivatives
        dC1 = []
        dC2 = []
        ddC1 = []
        ddC2 = []
        C_ov1 = np.zeros(N1)
        C_ov2 = np.zeros(N2)
        P_ov1 = np.zeros(N1)
        P_ov2 = np.zeros(N2)
        P_part1 = []
        P_part2 = []
        Mu1 = np.zeros(N1)
        Mu2 = np.zeros(N2)
        #T = self._Tg_init
        # Temperature gradient:
        dTg1 = c1_tmp._d@Tg1
        dTg2 = c2_tmp._d@Tg2
        ddTs1 =c1_tmp._dd@Ts1
        ddTs2 =c2_tmp._dd@Ts2

        # Concentration gradient
        # Pressure (overall&partial)
        # Viscosity
        for ii in range(n_comp):
            dC1.append(c1_tmp._d@C1[ii])
            dC2.append(c2_tmp._d@C2[ii])
            ddC1.append(c1_tmp._dd@C1[ii])
            ddC2.append(c2_tmp._dd@C2[ii])
            P_part1.append(C1[ii]*R_gas*Tg1/1E5) # in bar
            P_part2.append(C2[ii]*R_gas*Tg2/1E5) # in bar
            C_ov1 = C_ov1 + C1[ii]
            C_ov2 = C_ov2 + C2[ii]
            P_ov1 = P_ov1 + C1[ii]*R_gas*Tg1
            P_ov2 = P_ov2 + C2[ii]*R_gas*Tg2
            Mu1 = C1[ii]*c1_tmp._mu[ii]
            Mu2 = C2[ii]*c2_tmp._mu[ii]
        Mu1 = Mu1/C_ov1
        Mu2 = Mu2/C_ov2

        # Ergun equation
        v1,dv1 = Ergun(C1,Tg1,c1_tmp._M_m,Mu1,c1_tmp._D_p,epsi1,
        c1_tmp._d,c1_tmp._dd,c1_tmp._d_fo, N1)
        v2,dv2 = Ergun(C2,Tg2,c2_tmp._M_m,Mu2,c2_tmp._D_p,epsi2,
        c2_tmp._d,c2_tmp._dd,c2_tmp._d_fo, N2)
        
        # Solid phase concentration
        qsta1 = c1_tmp._iso(P_part1, Tg1) # partial pressure in bar
        qsta2 = c2_tmp._iso(P_part2, Tg2) # partial pressure in bar
        dqdt1 = []
        dqdt2 = []
        if c1_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii]*(qsta1[ii] - q1[ii])*a_surf1
                dqdt1.append(dqdt_tmp)
        elif c1_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii][0]*(qsta1[ii] - q1[ii])*a_surf1 + k_mass1[ii][1]*(qsta1[ii] - q1[ii])**2*a_surf1
                dqdt1.append(dqdt_tmp)
        if c2_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii]*(qsta2[ii] - q2[ii])*a_surf2
                dqdt2.append(dqdt_tmp)
        elif c2_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii][0]*(qsta2[ii] - q2[ii])*a_surf2 + k_mass2[ii][1]*(qsta2[ii] - q2[ii])**2*a_surf2
                dqdt2.append(dqdt_tmp)        

        # Valve equations (v_in and v_out)
        v_in1 = 0
        v_out2 = 0
        
        v_out1 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar
        v_in2 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar           
        
        # Gas phase concentration
        dCdt1 = []
        dCdt2 = []
        for ii in range(n_comp):
            dCdt_tmp = -v1*dC1[ii] -C1[ii]*dv1 + D_dis1[ii]*ddC1[ii] - (1-epsi1)/epsi1*rho_s1*dqdt1[ii]
            #dCdt_tmp[0] = +(v_in1*0 - v1[1]*C1[ii][0])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]               ######BOUNDARY C########
            dCdt_tmp[0] = +Cv_btw*(C_sta1[ii] - C1[ii][0]) - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]               ######BOUNDARY C########
            dCdt_tmp[-1]= +(v_out1+v1[-1])/2*(C1[ii][-2]-C1[ii][-1])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][-1] ######BOUNDARY C######## (KEY PROBLEM)
            dCdt1.append(dCdt_tmp)
        inout_ratio_mass=epsi1*c1_tmp._A/epsi2/c2_tmp._A
        for ii in range(n_comp):
            dCdt_tmp = -v2*dC2[ii] -C2[ii]*dv2 + D_dis2[ii]*ddC2[ii] - (1-epsi2)/epsi2*rho_s2*dqdt2[ii]
            dCdt_tmp[0] = +(v_in2*inout_ratio_mass*C1[ii][-1] - v2[1]*C2[ii][0])/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][0] ######BOUNDARY C########
            dCdt_tmp[-1]= +(v2[-1]*C2[ii][-2]- v_out2*C2[ii][-1])/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][-1]  ######BOUNDARY C########
            dCdt2.append(dCdt_tmp)

        # Temperature (gas)
        Cov_Cpg1 = np.zeros(N1) # Heat capacity (overall) J/K/m^3
        Cov_Cpg2 = np.zeros(N2) # Heat capacity (overall) J/K/m^3
        for ii in range(n_comp):
            Cov_Cpg1 = Cov_Cpg1 + Cpg1[ii]*C1[ii]
            Cov_Cpg2 = Cov_Cpg2 + Cpg2[ii]*C2[ii]
        dTgdt1 = -v1*dTg1 + h_heat1*a_surf1/epsi1*(Ts1 - Tg1)/Cov_Cpg1
        dTgdt2 = -v2*dTg2 + h_heat2*a_surf2/epsi2*(Ts2 - Tg2)/Cov_Cpg2
        for ii in range(n_comp):
            # column 1
            dTgdt1 = dTgdt1 - Cpg1[ii]*Tg1*D_dis1[ii]*ddC1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + Tg1*rho_s1*(1-epsi1)/epsi1*Cpg1[ii]*dqdt1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + h_ambi1*4/epsi1/D_col1*(T_ambi1 - Tg1)/Cov_Cpg1
            # column 2
            dTgdt2 = dTgdt2 - Cpg2[ii]*Tg2*D_dis2[ii]*ddC2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + Tg2*rho_s2*(1-epsi2)/epsi2*Cpg2[ii]*dqdt2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + h_ambi2*4/epsi2/D_col2*(T_ambi2 - Tg2)/Cov_Cpg2

        # column 1 dTgdt
        dTgdt1[0] = h_heat1*a_surf1/epsi1*(Ts1[0] - Tg1[0])/Cov_Cpg1[0]
        dTgdt1[0] = dTgdt1[0] + v_in1*(0-Tg1[0])/h1                         ######BOUNDARY C########: Tg[0]
        dTgdt1[-1] = h_heat1*a_surf1/epsi1*(Ts1[-1] - Tg1[-1])/Cov_Cpg1[-1]
        dTgdt1[-1] = dTgdt1[-1] + (v1[-1]+v_out1)/2*(Tg1[-2] - Tg1[-1])/h1  ######BOUNDARY C########: Tg[-1]
        # column 2 dTgdt
        inout_ratio_heat = Cov_Cpg1[-1]/Cov_Cpg2[0]
        dTgdt2[0] = h_heat2*a_surf2/epsi2*(Ts2[0] - Tg2[0])/Cov_Cpg2[0]
        dTgdt2[0] = dTgdt2[0] + (v_in2*inout_ratio_mass*Tg1[-1]  - v2[1]*Tg2[0])/h2*inout_ratio_heat  ######BOUNDARY C########
        dTgdt2[-1] = h_heat2*a_surf2/epsi2*(Ts2[-1] - Tg2[-1])/Cov_Cpg2[-1]
        dTgdt2[-1] = dTgdt2[-1] + (v2[-1]+v_out2)/2*(Tg2[-2] - 0)/h2        ######BOUNDARY C########

        # column 1&2 T boundary conditions 
        for ii in range(n_comp):
            dTgdt1[0] = dTgdt1[0] - Tg1[0]*Cpg1[ii]*dCdt1[ii][0]/Cov_Cpg1[0]
            dTgdt1[-1] = dTgdt1[-1] - Tg1[-1]*Cpg1[ii]*dCdt1[ii][-1]/Cov_Cpg1[-1]
            dTgdt2[0] = dTgdt2[0] - Tg2[0]*Cpg2[ii]*dCdt2[ii][0]/Cov_Cpg2[0]
            dTgdt2[-1] = dTgdt2[-1] - Tg2[-1]*Cpg2[ii]*dCdt2[ii][-1]/Cov_Cpg2[-1]
        dTsdt1 = (c1_tmp._k_cond*ddTs1+ h_heat1*a_surf1/(1-epsi1)*(Tg1-Ts1))/rho_s1/Cps1
        dTsdt2 = (c2_tmp._k_cond*ddTs2+ h_heat2*a_surf2/(1-epsi2)*(Tg2-Ts2))/rho_s2/Cps2
        for ii in range(n_comp):
            dTsdt1 = dTsdt1 + abs(dH1[ii])*dqdt1[ii]/Cps1
            dTsdt2 = dTsdt2 + abs(dH2[ii])*dqdt2[ii]/Cps2

        dydt_tmp1 = dCdt1+dqdt1+[dTgdt1] + [dTsdt1]
        dydt_tmp2 = dCdt2+dqdt2+[dTgdt2] + [dTsdt2]
        dydt1 = np.concatenate(dydt_tmp1)
        dydt2 = np.concatenate(dydt_tmp2)
        
        dydt = np.concatenate([dydt1,dydt2])
        
        # Check whether this converges
        #if np.max(np.abs(dydt1[0])) > 100:
        #    aaa  = 100/0
        #bool_list = np.abs(dydt) > y
        #if np.sum(bool_list) > 0:
        #    dydt = 1/2*dydt
        return dydt         
    C1_init = []
    C2_init = []
    for ii in range(n_comp):
        C1_init.append(c1_tmp._P_init*1E5/R_gas/T_mean*c1_tmp._y_init[ii])
        C2_init.append(c2_tmp._P_init*1E5/R_gas/T_mean*c2_tmp._y_init[ii])
    y0_tot_gas = np.concatenate(C1_init+C2_init)
    
    t_max_int_tneth = np.int32(np.floor(t_max/10))
    n_t_tenth = t_max_int_tneth*n_sec+ 1
    t_dom_tenth = np.linspace(0,t_max/10, n_t_tenth)
    if t_max_int_tneth < t_max/10:
        t_dom_tenth = np.concatenate((t_dom_tenth, [t_max/10]))
    
    #RUN1
    y_res = odeint(massmomeenbal_eq_gasonly, y0_tot_gas,t_dom_tenth)

    # Update concentration
    y0_tot[:N1*n_comp] = y_res[-1,:N1*n_comp]
    y0_tot[n_var_tot1:n_var_tot1+N2*n_comp] = y_res[-1,N1*n_comp:]
    # Update temperature
    y0_tot[n_var_tot1-2*N1:n_var_tot1-1*N1] = T_mean*np.ones(N1)
    y0_tot[-2*N2:-1*N2] = T_mean*np.ones(N2)
    
    #RUN2
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom)
    y_res1 = y_res[:,:n_var_tot1]
    y_res2 = y_res[:,n_var_tot1:]
    if flip1_later:
        y_res_flip1 = np.zeros_like(y_res1)
        for ii in range(n_comp*2+2):
             y_tmp = y_res1[:,ii*N1:(ii+1)*N1]
             y_res_flip1[:,ii*N1:(ii+1)*N1] = y_tmp@A_flip1
        y_res1 = y_res_flip1
    if flip2_later:
        y_res_flip2 = np.zeros_like(y_res2)
        for ii in range(n_comp*2+2):
            y_tmp = y_res2[:,ii*N2:(ii+1)*N2]
            y_res_flip2[:,ii*N2:(ii+1)*N2] = y_tmp@A_flip2
        y_res2 = y_res_flip2
    
    toc = time.time()/60 - tic
    c1_tmp._CPU_min = toc
    c2_tmp._CPU_min = toc
    if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
    column1._t = t_dom
    column2._t = t_dom
    if switch_later:
        column1._y = y_res2
        column2._y = y_res1
        column1._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        column2._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        return [[y_res2,column1._z, t_dom], [y_res1,column2._z, t_dom]]
    else:
        column1._y = y_res1
        column2._y = y_res2
        column1._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        column2._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        return [[y_res1,column1._z, t_dom], [y_res2,column2._z, t_dom]]

def step_P_eq_alt2(column1, column2, t_max,
n_sec=5, Cv_btw=0.1, valve_select = [1,1], CPUtime_print = False):
    """
    """
    tic = time.time() / 60 # in minute
    P_sum1 = np.mean(column1._P_init)
    P_sum2 = np.mean(column2._P_init)
    P_mean = (P_sum1+P_sum2)/2
    T_mean = (np.mean(column1._Tg_init)+np.mean(column2._Tg_init))/2
    if P_sum1 > P_sum2:
        c1_tmp = column1.copy()
        c2_tmp = column2.copy()
        val_sel = np.array(valve_select)
        switch_later = False
    else:
        c1_tmp = column2.copy()
        c2_tmp = column1.copy()
        val_sel = np.array([valve_select[0], valve_select[1]])
        switch_later = True
    if val_sel[0] == 0:
        A_flip1 = np.zeros([c1_tmp._N,c1_tmp._N])
        for ii in range(c1_tmp._N):
            A_flip1[ii, -1-ii] = 1
        c1_tmp._P_init =c1_tmp._P_init@A_flip1
        c1_tmp._Tg_init=c1_tmp._Tg_init@A_flip1
        c1_tmp._Ts_init=c1_tmp._Ts_init@A_flip1
        c1_tmp._y_init=c1_tmp._y_init@A_flip1
        c1_tmp._q_init=c1_tmp._q_init@A_flip1
        flip1_later = True
    else:
        flip1_later = False
    if val_sel[1]:
        A_flip2 = np.zeros([c2_tmp._N,c2_tmp._N])
        for ii in range(c2_tmp._N):
            A_flip2[ii, -1-ii] = 1
        c2_tmp._P_init =c2_tmp._P_init@A_flip2
        c2_tmp._Tg_init=c2_tmp._Tg_init@A_flip2
        c2_tmp._Ts_init=c2_tmp._Ts_init@A_flip2
        c2_tmp._y_init=c2_tmp._y_init@A_flip2
        c2_tmp._q_init=c2_tmp._q_init@A_flip2
        flip2_later=True
    else:
        flip2_later = False
    t_max_int = np.int32(np.floor(t_max))
    c1_tmp._n_sec = n_sec
    c2_tmp._n_sec = n_sec
    n_t = t_max_int*n_sec+ 1
    n_comp = column1._n_comp
    
    # Target pressure
    C_sta1 = np.zeros(n_comp)
    for ii in range(n_comp):
        C_sta1[ii] = c1_tmp._y_init[ii][0]*P_mean/R_gas/T_mean*1E5

    t_dom = np.linspace(0,t_max_int, n_t)
    column1._n_sec = n_sec
    column2._n_sec = n_sec
    if t_max_int < t_max:
        t_dom = np.concatenate((t_dom, [t_max]))
    N1 = c1_tmp._N
    N2 = c2_tmp._N
    h1 = c1_tmp._h
    h2 = c2_tmp._h
    epsi1 = c1_tmp._epsi
    epsi2 = c2_tmp._epsi

    a_surf1 = c1_tmp._a_surf
    a_surf2 = c2_tmp._a_surf
    # Mass
    D_dis1 = c1_tmp._D_disp
    D_dis2 = c2_tmp._D_disp

    k_mass1 = c1_tmp._k_mtc
    k_mass2 = c2_tmp._k_mtc

    # Heat
    dH1 = c1_tmp._dH
    dH2 = c2_tmp._dH
    Cpg1 = c1_tmp._Cp_g
    Cpg2 = c2_tmp._Cp_g
    
    Cps1 = c1_tmp._Cp_s
    Cps2 = c2_tmp._Cp_s

    h_heat1 = c1_tmp._h_heat
    h_heat2 = c2_tmp._h_heat

    h_ambi1 = c1_tmp._h_ambi
    h_ambi2 = c2_tmp._h_ambi

    T_ambi1 = c1_tmp._T_ambi
    T_ambi2 = c2_tmp._T_ambi

    # other parmeters
    rho_s1 = c1_tmp._rho_s
    rho_s2 = c2_tmp._rho_s
    D_col1 = np.sqrt(c1_tmp._A/np.pi)*2
    D_col2 = np.sqrt(c2_tmp._A/np.pi)*2

    n_var_tot1 = c1_tmp._N * (c1_tmp._n_comp+1)*2
    n_var_tot2 = c2_tmp._N * (c2_tmp._n_comp+1)*2
    
    # Initial conditions
    y01 = []
    y02 = []
    for ii in range(n_comp):
        C1_tmp = c1_tmp._P_init/R_gas/c1_tmp._Tg_init*c1_tmp._y_init[ii]*1E5    # in (mol/m^3)
        C2_tmp = c2_tmp._P_init/R_gas/c2_tmp._Tg_init*c2_tmp._y_init[ii]*1E5    # in (mol/m^3)
        y01.append(C1_tmp)
        y02.append(C2_tmp)
    for ii in range(n_comp):
        y01.append(c1_tmp._q_init[ii])
        y02.append(c2_tmp._q_init[ii])
    y01.append(c1_tmp._Tg_init)
    y02.append(c2_tmp._Tg_init)
    y01.append(c1_tmp._Ts_init)
    y02.append(c2_tmp._Ts_init)
    y01 = np.concatenate(y01)
    y02 = np.concatenate(y02)
    y0_tot = np.concatenate([y01,y02])

    # ODE function
    def massmomeenbal_eq(y,t):
        y1 = y[:n_var_tot1]
        y2 = y[n_var_tot1:]
        C1 = []
        C2 = []
        q1 = []
        q2 = []
        for ii in range(n_comp):
            C1.append(y1[ii*N1:(ii+1)*N1])
            C2.append(y2[ii*N2:(ii+1)*N2])
            q1.append(y1[n_comp*N1 + ii*N1 : n_comp*N1 + (ii+1)*N1])
            q2.append(y2[n_comp*N2 + ii*N2 : n_comp*N2 + (ii+1)*N2])
        Tg1 =y1[2*n_comp*N1 : 2*n_comp*N1 + N1 ]
        Tg2 =y2[2*n_comp*N2 : 2*n_comp*N2 + N2 ]
        Ts1 =y1[2*n_comp*N1 + N1 : 2*n_comp*N1 + 2*N1 ]
        Ts2 =y2[2*n_comp*N2 + N2 : 2*n_comp*N2 + 2*N2 ]

        # Derivatives
        dC1 = []
        dC2 = []
        ddC1 = []
        ddC2 = []
        C_ov1 = np.zeros(N1)
        C_ov2 = np.zeros(N2)
        P_ov1 = np.zeros(N1)
        P_ov2 = np.zeros(N2)
        P_part1 = []
        P_part2 = []
        Mu1 = np.zeros(N1)
        Mu2 = np.zeros(N2)
        #T = self._Tg_init
        # Temperature gradient:
        dTg1 = c1_tmp._d@Tg1
        dTg2 = c2_tmp._d@Tg2
        ddTs1 =c1_tmp._dd@Ts1
        ddTs2 =c2_tmp._dd@Ts2

        # Concentration gradient
        # Pressure (overall&partial)
        # Viscosity
        for ii in range(n_comp):
            dC1.append(c1_tmp._d@C1[ii])
            dC2.append(c2_tmp._d@C2[ii])
            ddC1.append(c1_tmp._dd@C1[ii])
            ddC2.append(c2_tmp._dd@C2[ii])
            P_part1.append(C1[ii]*R_gas*Tg1/1E5) # in bar
            P_part2.append(C2[ii]*R_gas*Tg2/1E5) # in bar
            C_ov1 = C_ov1 + C1[ii]
            C_ov2 = C_ov2 + C2[ii]
            P_ov1 = P_ov1 + C1[ii]*R_gas*Tg1
            P_ov2 = P_ov2 + C2[ii]*R_gas*Tg2
            Mu1 = C1[ii]*c1_tmp._mu[ii]
            Mu2 = C2[ii]*c2_tmp._mu[ii]
        Mu1 = Mu1/C_ov1
        Mu2 = Mu2/C_ov2

        # Ergun equation
        v1,dv1 = Ergun(C1,Tg1,c1_tmp._M_m,Mu1,c1_tmp._D_p,epsi1,
        c1_tmp._d,c1_tmp._dd,c1_tmp._d_fo, N1)
        v2,dv2 = Ergun(C2,Tg2,c2_tmp._M_m,Mu2,c2_tmp._D_p,epsi2,
        c2_tmp._d,c2_tmp._dd,c2_tmp._d_fo, N2)
        
        # Solid phase concentration
        qsta1 = c1_tmp._iso(P_part1, Tg1) # partial pressure in bar
        qsta2 = c2_tmp._iso(P_part2, Tg2) # partial pressure in bar
        dqdt1 = []
        dqdt2 = []
        if c1_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii]*(qsta1[ii] - q1[ii])*a_surf1
                dqdt1.append(dqdt_tmp)
        elif c1_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii][0]*(qsta1[ii] - q1[ii])*a_surf1 + k_mass1[ii][1]*(qsta1[ii] - q1[ii])**2*a_surf1
                dqdt1.append(dqdt_tmp)
        if c2_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii]*(qsta2[ii] - q2[ii])*a_surf2
                dqdt2.append(dqdt_tmp)
        elif c2_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii][0]*(qsta2[ii] - q2[ii])*a_surf2 + k_mass2[ii][1]*(qsta2[ii] - q2[ii])**2*a_surf2
                dqdt2.append(dqdt_tmp)        

        # Valve equations (v_in and v_out)
        v_in1 = 0
        v_out2 = 0
        
        v_out1 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar
        v_in2 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar           
        
        # Gas phase concentration
        dCdt1 = []
        dCdt2 = []
        for ii in range(n_comp):
            dCdt_tmp = -v1*dC1[ii] -C1[ii]*dv1 + D_dis1[ii]*ddC1[ii] - (1-epsi1)/epsi1*rho_s1*dqdt1[ii]
            #dCdt_tmp[0] = +(v_in1*0 - v1[1]*C1[ii][0])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]               ######BOUNDARY C########
            dCdt_tmp[0] = +(v_in1)/2*(C_sta1[ii] - C1[ii][0])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]               ######BOUNDARY C########
            dCdt_tmp[-1]= +(v_out1+v1[-1])/2*(C1[ii][-2]-C1[ii][-1])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][-1] ######BOUNDARY C######## (KEY PROBLEM)
            dCdt1.append(dCdt_tmp)
        inout_ratio_mass=epsi1*c1_tmp._A/epsi2/c2_tmp._A
        for ii in range(n_comp):
            dCdt_tmp = -v2*dC2[ii] -C2[ii]*dv2 + D_dis2[ii]*ddC2[ii] - (1-epsi2)/epsi2*rho_s2*dqdt2[ii]
            dCdt_tmp[0] = +(v_in2*inout_ratio_mass*C1[ii][-1] - v2[1]*C2[ii][0])/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][0] ######BOUNDARY C########
            dCdt_tmp[-1]= +(v2[-1]*C2[ii][-2]- v_out2*C2[ii][-1])/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][-1]  ######BOUNDARY C########
            dCdt2.append(dCdt_tmp)

        # Temperature (gas)
        Cov_Cpg1 = np.zeros(N1) # Heat capacity (overall) J/K/m^3
        Cov_Cpg2 = np.zeros(N2) # Heat capacity (overall) J/K/m^3
        for ii in range(n_comp):
            Cov_Cpg1 = Cov_Cpg1 + Cpg1[ii]*C1[ii]
            Cov_Cpg2 = Cov_Cpg2 + Cpg2[ii]*C2[ii]
        dTgdt1 = -v1*dTg1 + h_heat1*a_surf1/epsi1*(Ts1 - Tg1)/Cov_Cpg1
        dTgdt2 = -v2*dTg2 + h_heat2*a_surf2/epsi2*(Ts2 - Tg2)/Cov_Cpg2
        for ii in range(n_comp):
            # column 1
            dTgdt1 = dTgdt1 - Cpg1[ii]*Tg1*D_dis1[ii]*ddC1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + Tg1*rho_s1*(1-epsi1)/epsi1*Cpg1[ii]*dqdt1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + h_ambi1*4/epsi1/D_col1*(T_ambi1 - Tg1)/Cov_Cpg1
            # column 2
            dTgdt2 = dTgdt2 - Cpg2[ii]*Tg2*D_dis2[ii]*ddC2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + Tg2*rho_s2*(1-epsi2)/epsi2*Cpg2[ii]*dqdt2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + h_ambi2*4/epsi2/D_col2*(T_ambi2 - Tg2)/Cov_Cpg2

        # column 1 dTgdt
        dTgdt1[0] = h_heat1*a_surf1/epsi1*(Ts1[0] - Tg1[0])/Cov_Cpg1[0]
        dTgdt1[0] = dTgdt1[0] + v_in1*(0-Tg1[0])/h1                         ######BOUNDARY C########: Tg[0]
        dTgdt1[-1] = h_heat1*a_surf1/epsi1*(Ts1[-1] - Tg1[-1])/Cov_Cpg1[-1]
        dTgdt1[-1] = dTgdt1[-1] + (v1[-1]+v_out1)/2*(Tg1[-2] - Tg1[-1])/h1  ######BOUNDARY C########: Tg[-1]
        # column 2 dTgdt
        inout_ratio_heat = Cov_Cpg1[-1]/Cov_Cpg2[0]
        dTgdt2[0] = h_heat2*a_surf2/epsi2*(Ts2[0] - Tg2[0])/Cov_Cpg2[0]
        dTgdt2[0] = dTgdt2[0] + (v_in2*inout_ratio_mass*Tg1[-1]  - v2[1]*Tg2[0])/h2*inout_ratio_heat  ######BOUNDARY C########
        dTgdt2[-1] = h_heat2*a_surf2/epsi2*(Ts2[-1] - Tg2[-1])/Cov_Cpg2[-1]
        dTgdt2[-1] = dTgdt2[-1] + (v2[-1]+v_out2)/2*(Tg2[-2] - 0)/h2        ######BOUNDARY C########

        # column 1&2 T boundary conditions 
        for ii in range(n_comp):
            dTgdt1[0] = dTgdt1[0] - Tg1[0]*Cpg1[ii]*dCdt1[ii][0]/Cov_Cpg1[0]
            dTgdt1[-1] = dTgdt1[-1] - Tg1[-1]*Cpg1[ii]*dCdt1[ii][-1]/Cov_Cpg1[-1]
            dTgdt2[0] = dTgdt2[0] - Tg2[0]*Cpg2[ii]*dCdt2[ii][0]/Cov_Cpg2[0]
            dTgdt2[-1] = dTgdt2[-1] - Tg2[-1]*Cpg2[ii]*dCdt2[ii][-1]/Cov_Cpg2[-1]
        dTsdt1 = (c1_tmp._k_cond*ddTs1+ h_heat1*a_surf1/(1-epsi1)*(Tg1-Ts1))/rho_s1/Cps1
        dTsdt2 = (c2_tmp._k_cond*ddTs2+ h_heat2*a_surf2/(1-epsi2)*(Tg2-Ts2))/rho_s2/Cps2
        for ii in range(n_comp):
            dTsdt1 = dTsdt1 + abs(dH1[ii])*dqdt1[ii]/Cps1
            dTsdt2 = dTsdt2 + abs(dH2[ii])*dqdt2[ii]/Cps2
        for ii in range(n_comp):
            dCdt1[ii] = dCdt1[ii]
            dCdt2[ii] = dCdt2[ii]
        dydt_tmp1 = dCdt1+dqdt1+[dTgdt1] + [dTsdt1]
        dydt_tmp2 = dCdt2+dqdt2+[dTgdt2] + [dTsdt2]
        dydt1 = np.concatenate(dydt_tmp1)
        dydt2 = np.concatenate(dydt_tmp2)
        
        dydt = np.concatenate([dydt1,dydt2])
        
        # Check whether this converges
        #if np.max(np.abs(dydt1[0])) > 100:
        #    aaa  = 100/0
        #bool_list = np.abs(dydt) > y
        #if np.sum(bool_list) > 0:
        #    dydt = 1/2*dydt
        return dydt         
    
    #RUN
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom)
    y_res1 = y_res[:,:n_var_tot1]
    y_res2 = y_res[:,n_var_tot1:]
    if flip1_later:
        y_res_flip1 = np.zeros_like(y_res1)
        for ii in range(n_comp*2+2):
             y_tmp = y_res1[:,ii*N1:(ii+1)*N1]
             y_res_flip1[:,ii*N1:(ii+1)*N1] = y_tmp@A_flip1
        y_res1 = y_res_flip1
    if flip2_later:
        y_res_flip2 = np.zeros_like(y_res2)
        for ii in range(n_comp*2+2):
            y_tmp = y_res2[:,ii*N2:(ii+1)*N2]
            y_res_flip2[:,ii*N2:(ii+1)*N2] = y_tmp@A_flip2
        y_res2 = y_res_flip2
    
    toc = time.time()/60 - tic
    c1_tmp._CPU_min = toc
    c2_tmp._CPU_min = toc
    if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
    column1._t = t_dom
    column2._t = t_dom
    if switch_later:
        column1._y = y_res2
        column2._y = y_res1
        column1._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        column2._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        return [[y_res2,column1._z, t_dom], [y_res1,column2._z, t_dom]]
    else:
        column1._y = y_res1
        column2._y = y_res2
        column1._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        column2._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        return [[y_res1,column1._z, t_dom], [y_res2,column2._z, t_dom]]




def step_P_eq(column1, column2, t_max,
n_sec=5, Cv_btw=0.1, valve_select = [1,1], CPUtime_print = False):
    """
        
    """
    tic = time.time() / 60 # in minute
    P_sum1 = np.mean(column1._P_init)
    P_sum2 = np.mean(column2._P_init)
    if P_sum1 > P_sum2:
        c1_tmp = column1.copy()
        c2_tmp = column2.copy()
        val_sel = np.array(valve_select)
        switch_later = False
    else:
        c1_tmp = column2.copy()
        c2_tmp = column1.copy()
        val_sel = np.array([valve_select[0], valve_select[1]])
        switch_later = True
    if val_sel[0] == 0:
        A_flip1 = np.zeros([c1_tmp._N,c1_tmp._N])
        for ii in range(c1_tmp._N):
            A_flip1[ii, -1-ii] = 1
        c1_tmp._P_init =c1_tmp._P_init@A_flip1
        c1_tmp._Tg_init=c1_tmp._Tg_init@A_flip1
        c1_tmp._Ts_init=c1_tmp._Ts_init@A_flip1
        c1_tmp._y_init=c1_tmp._y_init@A_flip1
        c1_tmp._q_init=c1_tmp._q_init@A_flip1
        flip1_later = True
    else:
        flip1_later = False
    if val_sel[1]:
        A_flip2 = np.zeros([c2_tmp._N,c2_tmp._N])
        for ii in range(c2_tmp._N):
            A_flip2[ii, -1-ii] = 1
        c2_tmp._P_init =c2_tmp._P_init@A_flip2
        c2_tmp._Tg_init=c2_tmp._Tg_init@A_flip2
        c2_tmp._Ts_init=c2_tmp._Ts_init@A_flip2
        c2_tmp._y_init=c2_tmp._y_init@A_flip2
        c2_tmp._q_init=c2_tmp._q_init@A_flip2
        flip2_later=True
    else:
        flip2_later = False
    t_max_int = np.int32(np.floor(t_max))
    column1._n_sec = n_sec
    column2._n_sec = n_sec
    n_t = t_max_int*n_sec+ 1
    n_comp = column1._n_comp
    
    t_dom = np.linspace(0,t_max_int, n_t)
    if t_max_int < t_max:
        t_dom = np.concatenate((t_dom, [t_max]))
    N1 = c1_tmp._N
    N2 = c2_tmp._N
    h1 = c1_tmp._h
    h2 = c2_tmp._h
    epsi1 = c1_tmp._epsi
    epsi2 = c2_tmp._epsi

    a_surf1 = c1_tmp._a_surf
    a_surf2 = c2_tmp._a_surf
    # Mass
    D_dis1 = c1_tmp._D_disp
    D_dis2 = c2_tmp._D_disp

    k_mass1 = c1_tmp._k_mtc
    k_mass2 = c2_tmp._k_mtc

    # Heat
    dH1 = c1_tmp._dH
    dH2 = c2_tmp._dH
    Cpg1 = c1_tmp._Cp_g
    Cpg2 = c2_tmp._Cp_g
    
    Cps1 = c1_tmp._Cp_s
    Cps2 = c2_tmp._Cp_s

    h_heat1 = c1_tmp._h_heat
    h_heat2 = c2_tmp._h_heat

    h_ambi1 = c1_tmp._h_ambi
    h_ambi2 = c2_tmp._h_ambi

    T_ambi1 = c1_tmp._T_ambi
    T_ambi2 = c2_tmp._T_ambi

    # other parmeters
    rho_s1 = c1_tmp._rho_s
    rho_s2 = c2_tmp._rho_s
    D_col1 = np.sqrt(c1_tmp._A/np.pi)*2
    D_col2 = np.sqrt(c2_tmp._A/np.pi)*2

    n_var_tot1 = c1_tmp._N * (c1_tmp._n_comp+1)*2
    n_var_tot2 = c2_tmp._N * (c2_tmp._n_comp+1)*2
    
    # Initial conditions
    y01 = []
    y02 = []
    for ii in range(n_comp):
        C1_tmp = c1_tmp._P_init/R_gas/c1_tmp._Tg_init*c1_tmp._y_init[ii]*1E5    # in (mol/m^3)
        C2_tmp = c2_tmp._P_init/R_gas/c2_tmp._Tg_init*c2_tmp._y_init[ii]*1E5    # in (mol/m^3)
        y01.append(C1_tmp)
        y02.append(C2_tmp)
    for ii in range(n_comp):
        y01.append(c1_tmp._q_init[ii])
        y02.append(c2_tmp._q_init[ii])
    y01.append(c1_tmp._Tg_init)
    y02.append(c2_tmp._Tg_init)
    y01.append(c1_tmp._Ts_init)
    y02.append(c2_tmp._Ts_init)
    y01 = np.concatenate(y01)
    y02 = np.concatenate(y02)
    y0_tot = np.concatenate([y01,y02])

    # ODE function
    def massmomeenbal_eq(y,t):
        y1 = y[:n_var_tot1]
        y2 = y[n_var_tot1:]
        C1 = []
        C2 = []
        q1 = []
        q2 = []
        for ii in range(n_comp):
            C1.append(y1[ii*N1:(ii+1)*N1])
            C2.append(y2[ii*N2:(ii+1)*N2])
            q1.append(y1[n_comp*N1 + ii*N1 : n_comp*N1 + (ii+1)*N1])
            q2.append(y2[n_comp*N2 + ii*N2 : n_comp*N2 + (ii+1)*N2])
        Tg1 =y1[2*n_comp*N1 : 2*n_comp*N1 + N1 ]
        Tg2 =y2[2*n_comp*N2 : 2*n_comp*N2 + N2 ]
        Ts1 =y1[2*n_comp*N1 + N1 : 2*n_comp*N1 + 2*N1 ]
        Ts2 =y2[2*n_comp*N2 + N2 : 2*n_comp*N2 + 2*N2 ]

        # Derivatives
        dC1 = []
        dC2 = []
        ddC1 = []
        ddC2 = []
        C_ov1 = np.zeros(N1)
        C_ov2 = np.zeros(N2)
        P_ov1 = np.zeros(N1)
        P_ov2 = np.zeros(N2)
        P_part1 = []
        P_part2 = []
        Mu1 = np.zeros(N1)
        Mu2 = np.zeros(N2)
        #T = self._Tg_init
        # Temperature gradient:
        dTg1 = c1_tmp._d@Tg1
        dTg2 = c2_tmp._d@Tg2
        ddTs1 =c1_tmp._dd@Ts1
        ddTs2 =c2_tmp._dd@Ts2

        # Concentration gradient
        # Pressure (overall&partial)
        # Viscosity
        for ii in range(n_comp):
            dC1.append(c1_tmp._d@C1[ii])
            dC2.append(c2_tmp._d@C2[ii])
            ddC1.append(c1_tmp._dd@C1[ii])
            ddC2.append(c2_tmp._dd@C2[ii])
            P_part1.append(C1[ii]*R_gas*Tg1/1E5) # in bar
            P_part2.append(C2[ii]*R_gas*Tg2/1E5) # in bar
            C_ov1 = C_ov1 + C1[ii]
            C_ov2 = C_ov2 + C2[ii]
            P_ov1 = P_ov1 + C1[ii]*R_gas*Tg1
            P_ov2 = P_ov2 + C2[ii]*R_gas*Tg2
            Mu1 = C1[ii]*c1_tmp._mu[ii]
            Mu2 = C2[ii]*c2_tmp._mu[ii]
        Mu1 = Mu1/C_ov1
        Mu2 = Mu2/C_ov2

        # Ergun equation
        v1,dv1 = Ergun(C1,Tg1,c1_tmp._M_m,Mu1,c1_tmp._D_p,epsi1,
        c1_tmp._d,c1_tmp._dd,c1_tmp._d_fo, N1)
        v2,dv2 = Ergun(C2,Tg2,c2_tmp._M_m,Mu2,c2_tmp._D_p,epsi2,
        c2_tmp._d,c2_tmp._dd,c2_tmp._d_fo, N2)
        
        # Solid phase concentration
        qsta1 = c1_tmp._iso(P_part1, Tg1) # partial pressure in bar
        qsta2 = c2_tmp._iso(P_part2, Tg2) # partial pressure in bar
        dqdt1 = []
        dqdt2 = []
        if c1_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii]*(qsta1[ii] - q1[ii])*a_surf1
                dqdt_tmp = np.zeros(N1)
                dqdt1.append(dqdt_tmp)
        elif c1_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass1[ii][0]*(qsta1[ii] - q1[ii])*a_surf1 + k_mass1[ii][1]*(qsta1[ii] - q1[ii])**2*a_surf1
                dqdt_tmp = np.zeros(N1)
                dqdt1.append(dqdt_tmp)
        if c2_tmp._order_MTC == 1:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii]*(qsta2[ii] - q2[ii])*a_surf2
                dqdt_tmp = np.zeros(N2)
                dqdt2.append(dqdt_tmp)
        elif c2_tmp._order_MTC == 2:
            for ii in range(n_comp):
                dqdt_tmp = k_mass2[ii][0]*(qsta2[ii] - q2[ii])*a_surf2 + k_mass2[ii][1]*(qsta2[ii] - q2[ii])**2*a_surf2
                dqdt_tmp = np.zeros(N2)
                dqdt2.append(dqdt_tmp)        

        # Valve equations (v_in and v_out)
        v_in1 = 0
        v_out2 = 0
        
        v_out1 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar
        v_in2 = max(Cv_btw*(P_ov1[-1]/1E5 - P_ov2[0]/1E5), 0 )  # pressure in bar           
        
        # Gas phase concentration
        dCdt1 = []
        dCdt2 = []
        for ii in range(n_comp):
            dCdt_tmp = -v1*dC1[ii] -C1[ii]*dv1 + D_dis1[ii]*ddC1[ii] - (1-epsi1)/epsi1*rho_s1*dqdt1[ii]
            dCdt_tmp[0] = +(v_in1*0 - v1[1]*C1[ii][0])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][0]
            dCdt_tmp[-1]= +(v1[-1]*C1[ii][-2]- v_out1*C1[ii][-1])/h1 - (1-epsi1)/epsi1*rho_s1*dqdt1[ii][-1]
            dCdt1.append(dCdt_tmp)
        inout_ratio_mass=epsi1*c1_tmp._A/epsi2/c2_tmp._A
        for ii in range(n_comp):
            dCdt_tmp = -v2*dC2[ii] -C2[ii]*dv2 + D_dis2[ii]*ddC2[ii] - (1-epsi2)/epsi2*rho_s2*dqdt2[ii]
            dCdt_tmp[0] = +(v_in2*C1[ii][-1]*inout_ratio_mass - v2[1]*C2[ii][0])/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][0]
            dCdt_tmp[-1]= +(v2[-1]*C2[ii][-2]- v_out2*0)/h2 - (1-epsi2)/epsi2*rho_s2*dqdt2[ii][-1]
            dCdt2.append(dCdt_tmp)

        # Temperature (gas)
        Cov_Cpg1 = np.zeros(N1) # Heat capacity (overall) J/K/m^3
        Cov_Cpg2 = np.zeros(N2) # Heat capacity (overall) J/K/m^3
        for ii in range(n_comp):
            Cov_Cpg1 = Cov_Cpg1 + Cpg1[ii]*C1[ii]
            Cov_Cpg2 = Cov_Cpg2 + Cpg2[ii]*C2[ii]
        dTgdt1 = -v1*dTg1 + h_heat1*a_surf1/epsi1*(Ts1 - Tg1)/Cov_Cpg1
        dTgdt2 = -v2*dTg2 + h_heat2*a_surf2/epsi2*(Ts2 - Tg2)/Cov_Cpg2
        for ii in range(n_comp):
            # column 1
            dTgdt1 = dTgdt1 - Cpg1[ii]*Tg1*D_dis1[ii]*ddC1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + Tg1*rho_s1*(1-epsi1)/epsi1*Cpg1[ii]*dqdt1[ii]/Cov_Cpg1
            dTgdt1 = dTgdt1 + h_ambi1*4/epsi1/D_col1*(T_ambi1 - Tg1)/Cov_Cpg1
            # column 2
            dTgdt2 = dTgdt2 - Cpg2[ii]*Tg2*D_dis2[ii]*ddC2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + Tg2*rho_s2*(1-epsi2)/epsi2*Cpg2[ii]*dqdt2[ii]/Cov_Cpg2
            dTgdt2 = dTgdt2 + h_ambi2*4/epsi2/D_col2*(T_ambi2 - Tg2)/Cov_Cpg2

        # column 1 dTgdt
        dTgdt1[0] = h_heat1*a_surf1/epsi1*(Ts1[0] - Tg1[0])/Cov_Cpg1[0]
        dTgdt1[0] = dTgdt1[0] + (v_in1*0  - v1[1]*Tg1[0])/h1
        dTgdt1[-1] = h_heat1*a_surf1/epsi1*(Ts1[-1] - Tg1[-1])/Cov_Cpg1[-1]
        dTgdt1[-1] = dTgdt1[-1] + (v1[-1]*Tg1[-2]*Cov_Cpg1[-2]/Cov_Cpg1[-1] - v_out1*Tg1[-1])/h1
        # column 2 dTgdt
        inout_ratio_heat = Cov_Cpg1[-1]*epsi1*c1_tmp._A/Cov_Cpg2[0]/epsi2/c2_tmp._A
        dTgdt2[0] = h_heat2*a_surf2/epsi2*(Ts2[0] - Tg2[0])/Cov_Cpg2[0]
        dTgdt2[0] = dTgdt2[0] + (v_in2*Tg1[-1]*inout_ratio_heat  - v2[1]*Tg2[0])/h2
        dTgdt2[-1] = h_heat2*a_surf2/epsi2*(Ts2[-1] - 0)/Cov_Cpg2[-1]
        dTgdt2[-1] = dTgdt2[-1] + (v2[-1]*Tg2[-2]*Cov_Cpg2[-2]/Cov_Cpg2[-1] - v_out2*0)/h2
                
        # column 1&2 T boundary conditions 
        for ii in range(n_comp):
            dTgdt1[0] = dTgdt1[0] - Tg1[0]*Cpg1[ii]*dCdt1[ii][0]/Cov_Cpg1[0]
            dTgdt1[-1] = dTgdt1[-1] - Tg1[-1]*Cpg1[ii]*dCdt1[ii][-1]/Cov_Cpg1[-1]
            dTgdt2[0] = dTgdt2[0] - Tg2[0]*Cpg2[ii]*dCdt2[ii][0]/Cov_Cpg2[0]
            dTgdt2[-1] = dTgdt2[-1] - Tg2[-1]*Cpg2[ii]*dCdt2[ii][-1]/Cov_Cpg2[-1]
        dTsdt1 = (c1_tmp._k_cond*ddTs1+ h_heat1*a_surf1/(1-epsi1)*(Tg1-Ts1))/rho_s1/Cps1
        dTsdt2 = (c2_tmp._k_cond*ddTs2+ h_heat2*a_surf2/(1-epsi2)*(Tg2-Ts2))/rho_s2/Cps2
        for ii in range(n_comp):
            dTsdt1 = dTsdt1 + abs(dH1[ii])*dqdt1[ii]/Cps1
            dTsdt2 = dTsdt2 + abs(dH2[ii])*dqdt2[ii]/Cps2
        for ii in range(n_comp):
            dCdt1[ii] = dCdt1[ii]
            dCdt2[ii] = dCdt2[ii]
        dydt_tmp1 = dCdt1+dqdt1+[dTgdt1] + [dTsdt1]
        dydt_tmp2 = dCdt2+dqdt2+[dTgdt2] + [dTsdt2]
        dydt1 = np.concatenate(dydt_tmp1)
        dydt2 = np.concatenate(dydt_tmp2)
        
        dydt = np.concatenate([dydt1,dydt2])
        
        # Check whether this converges
        #if np.max(np.abs(dydt1[0])) > 100:
        #    aaa  = 100/0
        #bool_list = np.abs(dydt) > y
        #if np.sum(bool_list) > 0:
        #    dydt = 1/2*dydt
        return dydt         

    #RUN
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom)
    y_res1 = y_res[:,:n_var_tot1]
    y_res2 = y_res[:,n_var_tot1:]
    C_sum = 0
    for ii in range(n_comp):
        C_sum = C_sum + y_res1[-1,ii*N1+2]
        C_sum = C_sum + y_res2[-1,ii*N2+2]
    if C_sum < 0.5:
        y_res12 = step_P_eq_alt1(column1,column2, t_max, n_sec = n_sec,
        Cv_btw = Cv_btw, valve_select = valve_select)
        toc = time.time()/60 - tic
        C_sum = 0
        y_res1 = y_res[:,:n_var_tot1]
        y_res2 = y_res[:,n_var_tot1:]
        for ii in range(n_comp):
            C_sum = C_sum + y_res1[-1,ii*N1+2]
            C_sum = C_sum + y_res2[-1,ii*N2+2]
        return y_res12
        if C_sum < 0.5:
            y_res12 = step_P_eq_alt2(column1,column2, t_max, n_sec = n_sec,
            Cv_btw = Cv_btw, valve_select = valve_select)
            toc = time.time()/60 - tic
            column1._CPU_min = toc
            column2._CPU_min = toc
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))
            return y_res12
        else:
            column1._CPU_min = toc
            column2._CPU_min = toc
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))
            return y_res12
    if flip1_later:
        y_res_flip1 = np.zeros_like(y_res1)
        for ii in range(n_comp*2+2):
             y_tmp = y_res1[:,ii*N1:(ii+1)*N1]
             y_res_flip1[:,ii*N1:(ii+1)*N1] = y_tmp@A_flip1
        y_res1 = y_res_flip1
    if flip2_later:
        y_res_flip2 = np.zeros_like(y_res2)
        for ii in range(n_comp*2+2):
            y_tmp = y_res2[:,ii*N2:(ii+1)*N2]
            y_res_flip2[:,ii*N2:(ii+1)*N2] = y_tmp@A_flip2
        y_res2 = y_res_flip2
    
    toc = time.time()/60 - tic
    c1_tmp._CPU_min = toc
    c2_tmp._CPU_min = toc
    if CPUtime_print:
            print('Simulation of this step is completed.')
            print('This took {0:9.3f} mins to run. \n'.format(toc))       
    column1._t = t_dom
    column2._t = t_dom
    if switch_later:
        column1._y = y_res2
        column2._y = y_res1
        column1._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        column2._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        return [[y_res2,column1._z, t_dom], [y_res1,column2._z, t_dom]]
    else:
        column1._y = y_res1
        column2._y = y_res2
        column1._Tg_res = y_res1[:,n_comp*2*N1 : n_comp*2*N1+N1]
        column2._Tg_res = y_res2[:,n_comp*2*N2 : n_comp*2*N2+N2]
        return [[y_res1,column1._z, t_dom], [y_res2,column2._z, t_dom]]
    
# %% When only this code is run (name == main)
if __name__ == '__main__':
    N = 11
    A_cros = 0.031416
    L = 1
    c1 = column(L,A_cros, n_component = 2,N_node = N)
    """
    """
    ## Adsorbent
    isopar1 = [3.0, 1]
    isopar2 = [1.0, 0.5]
    def iso_fn_test(P,T):
        b1 = isopar1[1]*np.exp(30E3/8.3145*(1/T-1/300))
        b2 = isopar2[1]*np.exp(20E3/8.3145*(1/T-1/300))
        denom = 1 + b1*P[0] + b2*P[1]
        numor0 = isopar1[0]*b1*P[0]
        numor1 = isopar2[0]*b2*P[1]
        q_return = [numor0/denom, numor1/denom]
        return q_return
 
    epsi_test = 0.4         # macroscopic void fraction (m^3/m^3)
    D_particle_dia = 0.01   # particle diameter (m)
    rho_s_test = 1100       # solid density (kg/mol)
    c1.adsorbent_info(iso_fn_test,epsi_test,D_particle_dia, rho_s_test)
 
    M_m_test  = [0.044, 0.028]      ## molar mass    (kg/mol)
    mu_test = [1.47E-5, 1.74E-5]    ## gas viscosity (Pa sec)
    c1.gas_prop_info(M_m_test, mu_test)
 
    ## Mass transfer coefficients
    D_dis_test = [1E-6, 1E-6]   # m^2/sec
    k_MTC = [0.0002, 0.0002]    # m/sec
    a_surf = 400                # m^2/m^3
    c1.mass_trans_info(k_MTC, a_surf, D_dis_test)

    ## Thermal properties
    Del_H = [30E3, 20E3]    # J/mol
    Cp_s = 935              # J/kg/K
    Cp_g = [37.22, 29.15]   # J/mol/K
    h_heat = 100            # J/sec/m^2/K
    c1.thermal_info(Del_H,Cp_s,Cp_g,h_heat,)

    ## Boundary condition
    Pin_test = 1.5      # inlet pressure (bar)
    yin_test = [1, 0]   # inlet composition (mol/mol)
    Tin_test = 300      # inlet temperature (K)
    Q_in_test = 0.2*0.031416*0.3  # volumetric flowrate (m^3/sec)
    Cvin_test = 1E-1    # inlet valve constant (m/sec/bar)
 
    Pout_test = 1       # outlet pressure (bar)
    Cvout_test = 2E-2   # outlet valve constant (m/sec/bar)
    c1.boundaryC_info(Pout_test,Pin_test,Tin_test,yin_test,
    Cvin_test,Cvout_test,Q_in_test,assigned_v_option = False)
 
    #c1.boundaryC_info(Pout_test, Pin_test,Tin_test,yin_test,Cvin_test)
 
    ## Initial condition
    P_init = 1*np.ones(N)                   # initial pressure (bar)
    y_init = [0*np.ones(N), 1*np.ones(N)]   # initial composition (mol/mol)
    T_init = 300*np.ones(N)                 # initial temperature (K)
    q_init = iso_fn_test([P_init*y_init[0],
    P_init*y_init[1]],T_init)               # initial uptake
    c1.initialC_info(P_init, T_init, T_init,y_init,q_init)
    
    ## print here
    print(c1)
    c1.run_mamoen(500,CPUtime_print = True)
    c1.Graph(
        100,0,loc = [0.8,0.85], 
        yaxis_label=r'C$_{1}$ (mol/m$^{3}$)',
        figsize = [8.5,5])
    c1.Graph_P(
        100, loc = [1.1,0.85],
        figsize = [8.5,5],)
    c2 = c1.copy()
    c1.next_init()

    c1.boundaryC_info(Pout_test,Pin_test,Tin_test,yin_test,
    0.0*Cvin_test,Cvout_test,Q_in_test,False, foward_flow_direction=True)
    c2.boundaryC_info(Pout_test,Pin_test,Tin_test,yin_test,
    0.0*Cvin_test,0,Q_in_test,False, foward_flow_direction=True)

    step_P_eq(
        c1,c2,100,n_sec = 50,
        valve_select = [1,1],
        Cv_btw=0.05,CPUtime_print=True)
    plt.show()
    Legend_loc = [1.15, 0.9]
    c1.Graph_P(10, loc = Legend_loc)
    c1.Graph(10,2, loc =Legend_loc)
    c1.Graph(10,3, loc =Legend_loc)
    c2.Graph_P(10, loc =Legend_loc)
    c2.Graph(10,2, loc =Legend_loc)
    c2.Graph(10,3, loc = Legend_loc)
    plt.show( )
