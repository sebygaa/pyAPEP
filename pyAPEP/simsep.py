
# %% Import packages
import numpy as np
#from numpy.lib.function_base import _parse_input_dimensions
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
#from scipy.integrate import solve_ivp
 
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
    dP = d_fo@P
    #ddP = dd@P

    Vs_Vg = (1-epsi_void)/epsi_void
    A =1.75*rho_g/D_particle*Vs_Vg
    B = 150*mu_vis/D_particle**2*Vs_Vg**2
    C = dP
 
    ind_posi = B**2-4*A*C >= 0
    ind_nega = ind_posi == False
    v_pos = (-B[ind_posi]+ np.sqrt(B[ind_posi]**2-4*A[ind_posi]*C[ind_posi]))/(2*A[ind_posi])
    v_neg = (B[ind_nega] - np.sqrt(B[ind_nega]**2+4*A[ind_nega]*C[ind_nega]))/(2*A[ind_nega])
    # v_neg = 0 
    v_return = np.zeros(N)
    v_return[ind_posi] = v_pos
    v_return[ind_nega] = v_neg

    #AA = -ddP/Vs_Vg
    #BB = 3.5*rho_g/D_particle*v_return
    #CC = 150*mu_vis/D_particle**2
    #dv_pos = AA[ind_posi]/(BB[ind_posi]+CC[ind_posi])
    #dv_neg = AA[ind_nega]/(BB[ind_nega]-CC[ind_nega])
    # dv_pos = -epsi_void*ddP[ind_posi]/(3.5*rho_g[ind_posi]/D_particle*v_return[ind_posi]+150*mu_vis[ind_posi]/D_particle**2)/(1-epsi_void)
    
    # dv_neg = -epsi_void*ddP[ind_nega]/(3.5*rho_g[ind_nega]/D_particle*v_return[ind_nega]-150*mu_vis[ind_posi]/D_particle**2)/(1-epsi_void)
    # dv_return = np.zeros(N)
    # dv_return[ind_posi] = dv_pos
    # dv_return[ind_nega] = dv_neg
    dv_return = d@v_return
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
            y_new_tmp[y_new_tmp < 0] = 0
            y_return.append(y_new_tmp)
    elif len(y_raw.shape) == 1:
        yy = y_raw
        fn_tmp = interp1d(z_raw, yy, kind = 'cubic')
        z_new = np.linspace(z_raw[0], z_raw[-1],N_new)
        y_return = fn_tmp(z_new)
        y_return[y_return < 0] = 0
    elif len(y_raw.shape) == 2:
        yy = y_raw[-1,:]
        fn_tmp = interp1d(z_raw, yy, kind = 'cubic')
        z_new = np.linspace(z_raw[0], z_raw[-1],N_new)
        y_return = fn_tmp(z_new)
        y_return[y_return < 0] = 0
    else:
        print('Input should be 1d or 2d array.')
        return None
    
    return y_return

# %% GaODE: NewODE function
def gaode(dy_fun, y0, t, args= None):
#    if np.isscalar(t):
#        t_domain = np.linspace(0,t, 10001, dtype=np.float64)
#    else:
#        t_domain = np.array(t[:], dtype = np.float64)
    t_domain = np.array(t[:], dtype = np.float64)
    y_res = []
    dt_arr = t_domain[1:] - t_domain[:-1]

    N = len(y0)
    tt_prev = t_domain[0]
    y_tmp = np.array(y0, dtype = np.float64)
    y_res.append(y_tmp)
    if args == None:
        for tt, dtt in zip(t_domain[:-1], dt_arr):
            dy_tmp = dy_fun(y_tmp, tt)
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
#            if tt%10 == 1:
#                print(y_tmp_new, y_tmp)
        y_res_arr = np.array(y_res, dtype = np.float64)
    else:
        for tt, dtt in zip(t_domain[1:], dt_arr):
            dy_tmp = dy_fun(y_tmp, tt, *args)
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
        y_res_arr = np.array(y_res, dtype=object)
    
    return y_res_arr


# %% Column class
class column:
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
        self.is_BASIS = False
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
                Mu = Mu + C[ii]*self._mu[ii]
            Mu = Mu/C_ov
            #Mu = np.mean(self._mu)*np.ones_like(self._N)
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
                #dCdt_tmp[0] = v_in*(C_sta[ii]-C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v_out+v[-1])/2*(C[ii][-2]- C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = (v_in*C_sta[ii]-v[0]*C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-2]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
#                dv0 = (v[0] - v_in)/h
#                dC0 = (C[ii][0] - C_sta[ii])/h
#
#                dvL = (v_out-v[-2])/h
#                dCL = (C[ii][-1] - C[ii][-2])/h
#                dCdt_tmp[0] = -v[0]*dC0 - C[ii][0]*dv0 - (1-epsi)/epsi*rho_s*dqdt[ii][0]
#                dCdt_tmp[-1]= -v_out*dCL-C[ii][-1]*dvL - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3

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
        y_result = odeint(massmomebal,y0,t_dom,rtol=1e-6, atol=1e-9)
        
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

    def run_ma(self, t_max, n_sec = 5, CPUtime_print = False):
        tic = time.time()/60
        t_max_int = np.int32(np.floor(t_max), )
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        n_comp = self._n_comp
        D_dif = self._D_disp
        C_sta = []
        for ii in range(n_comp):
            C_sta.append(self._y_in[ii]*self._P_in/R_gas/self._T_in*1E5)
        Cbound = C_sta
        k = np.array(self._k_mtc)*self._a_surf
        T_const = self._T_in
        P_ov = self._P_in
        rho = self._rho_s
        epsi = self._epsi
        u0 = self._Q_in/epsi/self._A
        N = self._N
        
        # Difference matrix
        d = self._d
        dd = self._dd

        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        def massonly(y,t):
            C = []
            q = []
            dC = []
            ddC = []
            for ii in range(n_comp):
                C.append(y[ii*N:(ii+1)*N])
                q.append(y[n_comp*N + ii*N : n_comp*N + (ii+1)*N])
            
            q_sta = []
            P_part = []
            C_ov = np.zeros([N,])
            for ii in range(n_comp):
                dC.append(d@C[ii])
                ddC.append(dd@C[ii])
                C_ov = C_ov + C[ii]
            yfrac = []
            for ii in range(n_comp):
                yfrac_tmp = C[ii]/C_ov
                P_part_tmp = yfrac_tmp*P_ov
                yfrac.append(yfrac_tmp)
                P_part.append(yfrac_tmp*P_ov) # in bar
            q_sta = self._iso(P_part ,T_const)
            #print(q_sta)
            dqdt = []
            for ii in range(n_comp):
                dqdt_tmp = k[ii]*(q_sta[ii] - q[ii])
                dqdt.append(dqdt_tmp)

            v = u0
            dCdt = []
            dydt_pr = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] + D_dif[ii]*ddC[ii] - (1-epsi)/epsi*rho*dqdt[ii]
                dCdt_tmp[0] = (v*Cbound[ii] -v*C[ii][0])*d[2,2] - (1-epsi)/epsi*rho*dqdt[ii][0]
                dCdt.append(dCdt_tmp)
                dydt_pr.append(dCdt_tmp)
            for ii in range(n_comp):
                dydt_pr.append(dqdt[ii])
            dydt = np.concatenate(dydt_pr)
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
        y_result = odeint(massonly, y0, t_dom, rtol=1e-6, atol=1e-9)
        
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


    def run_ma_POD(self, t_max, n_sec = 5, N_basis = 20, U_basis = None, CPUtime_print = False):
        tic = time.time()/60
        if (type(None)==type(U_basis)):
            if self.is_BASIS:
                U_basis = self._U_basis
            else:
                print("Please use 'find_basis' first to find the basis!")
                return None
        elif (type(U_basis) == type(np.array([1,2,3,]))):
            if len(U_basis) <= 2:
                if self.is_BASIS:
                    U_basis = self._U_basis
                else:
                    print("Please use 'find_basis' first to find the basis!")
                    return None
        U_cut = U_basis[:,:N_basis]
        
        tic = time.time()/60
        t_max_int = np.int32(np.floor(t_max), )
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        n_comp = self._n_comp
        D_dif = self._D_disp
        C_sta = []
        for ii in range(n_comp):
            C_sta.append(self._y_in[ii]*self._P_in/R_gas/self._T_in*1E5)
        Cbound = C_sta

        k = np.array(self._k_mtc)*self._a_surf
        T_const = self._T_in
        P_ov = self._P_in
        rho = self._rho_s
        epsi = self._epsi
        u0 = self._Q_in/epsi/self._A
        N = self._N
        
        # Difference matrix
        d = self._d
        dd = self._dd

        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        def massonly(aa,t):
            y = U_cut@aa
            C = []
            q = []
            dC = []
            ddC = []
            for ii in range(n_comp):
                C.append(y[ii*N:(ii+1)*N])
                q.append(y[n_comp*N + ii*N : n_comp*N + (ii+1)*N])
            
            q_sta = []
            P_part = []
            C_ov = np.zeros([N,])
            for ii in range(n_comp):
                dC.append(d@C[ii])
                ddC.append(dd@C[ii])
                C_ov = C_ov + C[ii]
            yfrac = []
            for ii in range(n_comp):
                yfrac_tmp = C[ii]/C_ov
                P_part_tmp = yfrac_tmp*P_ov
                yfrac.append(yfrac_tmp)
                P_part.append(yfrac_tmp*P_ov) # in bar
            q_sta = self._iso(P_part ,T_const)
            #print(q_sta)
            dqdt = []
            for ii in range(n_comp):
                dqdt_tmp = k[ii]*(q_sta[ii] - q[ii])
                dqdt.append(dqdt_tmp)

            v = u0
            dCdt = []
            dydt_pr = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] + D_dif[ii]*ddC[ii] - (1-epsi)/epsi*rho*dqdt[ii]
                dCdt_tmp[0] = (v*Cbound[ii] -v*C[ii][0])*d[2,2] - (1-epsi)/epsi*rho*dqdt[ii][0]
                dCdt.append(dCdt_tmp)
                dydt_pr.append(dCdt_tmp)
            for ii in range(n_comp):
                dydt_pr.append(dqdt[ii])
            dydt = np.concatenate(dydt_pr)
            daadt = U_cut.T@dydt
            return daadt
        
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
        aa0 = U_cut.T@y0

        #RUN
        aa_result = odeint(massonly, aa0, t_dom, rtol=1e-6, atol=1e-9)
        y_result = aa_result@U_cut.T
        
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



## Run mass & momentum balance equations
    def run_mamo_POD(self, t_max, n_sec = 5, N_basis = 20, U_basis = None, CPUtime_print = False):
        tic = time.time()/60
        if (type(None)==type(U_basis)):
            if self.is_BASIS:
                U_basis = self._U_basis
            else:
                print("Please use 'find_basis' first to find the basis!")
                return None
        elif (type(U_basis) == type(np.array([1,2,3,]))):
            if len(U_basis) <= 2:
                if self.is_BASIS:
                    U_basis = self._U_basis
                else:
                    print("Please use 'find_basis' first to find the basis!")
                    return None
                
        U_cut = U_basis[:, :N_basis]
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
        def massmomebal(aa,t):
            y = U_cut@aa
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
                Mu = Mu + C[ii]*self._mu[ii]
            Mu = Mu/C_ov
            #Mu = np.mean(self._mu)*np.ones_like(self._N)
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
                #dCdt_tmp[0] = v_in*(C_sta[ii]-C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v_out+v[-1])/2*(C[ii][-2]- C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = (v_in*C_sta[ii]-v[0]*C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-2]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
#                dv0 = (v[0] - v_in)/h
#                dC0 = (C[ii][0] - C_sta[ii])/h
#
#                dvL = (v_out-v[-2])/h
#                dCL = (C[ii][-1] - C[ii][-2])/h
#                dCdt_tmp[0] = -v[0]*dC0 - C[ii][0]*dv0 - (1-epsi)/epsi*rho_s*dqdt[ii][0]
#                dCdt_tmp[-1]= -v_out*dCL-C[ii][-1]*dvL - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3

                dCdt.append(dCdt_tmp)
 
            dydt_tmp = dCdt+dqdt
            dydt = np.concatenate(dydt_tmp)
            daadt = U_cut.T@dydt
            return daadt
        
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
        
        # Change the basis (coordinate)
        aa0 = U_cut.T@y0

        #RUN
        aa_result = odeint(massmomebal,aa0,t_dom, rtol=1e-6, atol=1e-9)
        
        # Rolling back to Original coordinate
        y_result = aa_result@U_cut.T

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
                Mu = Mu + C[ii]*self._mu[ii]
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
                #dCdt_tmp[0] = +v_in*(C_sta[ii] - C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v_out+v[-1])/2*(C[ii][-2]- C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = (v_in*C_sta[ii]-v[0]*C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-2]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
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
        y_result = odeint(massmomeenerbal_alt,y0,t_dom, rtol=1e-6, atol=1e-9)
        
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
        
        def massmomeenerbal(y, t):
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
                Mu = Mu + C[ii]*self._mu[ii]
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
                v_in = self._Cv_in*(self._P_in - P_ov[0]/1E5)  # pressure in bar           
            v_out = max(self._Cv_out*(P_ov[-1]/1E5 - self._P_out), 0 )  # pressure in bar
            
            # Gas phase concentration
            dCdt = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] -C[ii]*dv + D_dis[ii]*ddC[ii] - (1-epsi)/epsi*rho_s*dqdt[ii]
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]    
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v[-1]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = (v_in*C_sta[ii]-v[0]*C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-2]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
                dv0 = (v[0] - v_in)/h
                dC0 = (C[ii][0] - C_sta[ii])/h

                dvL = (v_out-v[-2])/h
                dCL = (C[ii][-1] - C[ii][-2])/h
                dCdt_tmp[0] = -v[0]*dC0 - C[ii][0]*dv0 - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= -v_out*dCL-C[ii][-1]*dvL - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
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
            dTgdt[0] = dTgdt[0] + (v_in*self._T_in*Cov_Cpg_in/Cov_Cpg[0] - v[0]*Tg[0])/h
            dTgdt[-1] = h_heat*a_surf/epsi*(Ts[-1] - Tg[-1])/Cov_Cpg[-1]
            #dTgdt[-1] = dTgdt[-1] + (v[-1]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            dTgdt[-1] = dTgdt[-1] + (v[-2]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
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
        y_result = odeint(massmomeenerbal,y0, t_dom, rtol=1e-6, atol=1e-9)
        #y_ivp = solve_ivp(massmomeenerbal,t_dom, y0,method = 'BDF')
        #y_result = y_ivp.y
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
            return y_result, self._z, t_dom
        
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

    def find_basis(self, show_graph = False, show_n = 20):
        try:
            self._y
        except:
            print("You have to run the simulation, first!")
            return
        U, Sig, Vt = np.linalg.svd(self._y.T)
        self.is_BASIS = True
        self._U_basis = U
        if show_graph:
            if show_n == -1:
                fig = plt.figure()
                plt.bar(np.arange(1,len(U)), 
                         Sig,
                         color = 'k')
                plt.xlabel("Principle components")
                plt.ylabel("Singular value")
            else:
                fig = plt.figure()
                plt.bar(np.arange(1,show_n+1), 
                        Sig[:show_n],
                        color = 'k')
                plt.xlabel("Principle components")
                plt.ylabel("Singular value")
            return U, Sig, Vt, fig
        
        return U, Sig, Vt
        

    def run_mamoen_POD(self, t_max, n_sec = 5, N_basis = 20, U_basis = None, CPUtime_print = False):
        if (type(None)==type(U_basis)):
            if self.is_BASIS:
                U_basis = self._U_basis
            else:
                print("Please use 'find_basis' first to find the basis!")
                return None
        elif (type(U_basis) == type(np.array([1,2,3,]))):
            if len(U_basis) <= 2:
                if self.is_BASIS:
                    U_basis = self._U_basis
                else:
                    print("Please use 'find_basis' first to find the basis!")
                    return None
        U_cut = U_basis[:,:N_basis]
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
        
        def massmomeenerbal(aa, t):
            y = U_cut@aa
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
                Mu = Mu + C[ii]*self._mu[ii]
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
                v_in = self._Cv_in*(self._P_in - P_ov[0]/1E5)  # pressure in bar           
            v_out = max(self._Cv_out*(P_ov[-1]/1E5 - self._P_out), 0 )  # pressure in bar
            
            # Gas phase concentration
            dCdt = []
            for ii in range(n_comp):
                dCdt_tmp = -v*dC[ii] -C[ii]*dv + D_dis[ii]*ddC[ii] - (1-epsi)/epsi*rho_s*dqdt[ii]
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]    
                #dCdt_tmp[0] = +(v_in*C_sta[ii] - v[1]*C[ii][0])/h - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                #dCdt_tmp[-1]= +(v[-1]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
                dCdt_tmp[0] = (v_in*C_sta[ii]-v[0]*C[ii][0])/h- (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= +(v[-2]*C[ii][-2]- v_out*C[ii][-1])/h - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
                dv0 = (v[0] - v_in)/h
                dC0 = (C[ii][0] - C_sta[ii])/h

                dvL = (v_out-v[-2])/h
                dCL = (C[ii][-1] - C[ii][-2])/h
                dCdt_tmp[0] = -v[0]*dC0 - C[ii][0]*dv0 - (1-epsi)/epsi*rho_s*dqdt[ii][0]
                dCdt_tmp[-1]= -v_out*dCL-C[ii][-1]*dvL - (1-epsi)/epsi*rho_s*dqdt[ii][-1]
###################################################3
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
            dTgdt[0] = dTgdt[0] + (v_in*self._T_in*Cov_Cpg_in/Cov_Cpg[0] - v[0]*Tg[0])/h
            dTgdt[-1] = h_heat*a_surf/epsi*(Ts[-1] - Tg[-1])/Cov_Cpg[-1]
            #dTgdt[-1] = dTgdt[-1] + (v[-1]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            dTgdt[-1] = dTgdt[-1] + (v[-2]*Tg[-2]*Cov_Cpg[-2]/Cov_Cpg[-1] - v_out*Tg[-1])/h
            for ii in range(n_comp):
                dTgdt[0] = dTgdt[0] - Tg[0]*Cpg[ii]*dCdt[ii][0]/Cov_Cpg[0]
                dTgdt[-1] = dTgdt[-1] - Tg[-1]*Cpg[ii]*dCdt[ii][-1]/Cov_Cpg[-1]
            dTsdt = (self._k_cond*ddTs+ h_heat*a_surf/(1-epsi)*(Tg-Ts))/self._rho_s/Cps
            for ii in range(n_comp):
                dTsdt = dTsdt + abs(dH[ii])*dqdt[ii]/Cps
            
            dydt_tmp = dCdt+dqdt+[dTgdt] + [dTsdt]
            dydt = np.concatenate(dydt_tmp)
            daadt = U_cut.T@dydt
            return daadt
        
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
        aa0 = U_cut.T@y0
        #RUN
        a_result = odeint(massmomeenerbal, aa0, t_dom, rtol=1e-6, atol=1e-9)
        y_result = a_result@U_cut.T
        #y_ivp = solve_ivp(massmomeenerbal,t_dom, y0,method = 'BDF')
        #y_result = y_ivp.y
        C_sum = 0
        '''
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
            return y_result, self._z, t_dom
        '''
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
    ## Copy the column
    def copy(self):
        import copy
        self_new = copy.deepcopy(self)
        return self_new
    
    ##### Linear velocity assumptions #####
    ## Run mass balance only with linear velocity assumption ##
    def run_ma_linvel(self, t_max, n_sec = 5, CPUtime_print = False):
        # t domain
        tic = time.time()/60 # in minute
        t_max_int = np.int32(np.floor(t_max))
        self._n_sec = n_sec
        n_t = t_max_int*n_sec+ 1
        
        t_dom = np.linspace(0,t_max_int, n_t)
        if t_max_int < t_max:
            t_dom = np.concatenate((t_dom, [t_max]))
        
        # z axis
        N = self._N
        h = self._h
        L = self._L
        z = np.linspace(0,L,N)
        # Geometry
        A_cros = self._A
        V_tot = L*self._A
        epsi = self._epsi
        dp = self._D_p
        a_surf = self._a_surf
        rho_s = self._rho_s
        # n_comp
        n_comp = self._n_comp
        # Initial Conditions
        y_av = []
        mu_av = 0
        for ii in range(n_comp):
            y_av_tmp = np.mean(self._y_init[ii])
            mu_av = mu_av + y_av_tmp*self._mu[ii]
            y_av.append(y_av_tmp)
        # Boundary (feed) conditions
        P_feed = self._P_in
        P_out = self._P_out
        T_feed = self._T_in
        y_feed = self._y_in      
        # Three diff cases
        # Case 1: Flowing through (both open valve)
        if (self._Cv_in > 1E-10) and (self._Cv_out > 1E-10):
            ########### IF "BACKWARD" ???? ########
            # Velocity (constant)
            u_feed = self._Q_in/epsi/self._A
            P_L = P_out + self._Q_in/self._Cv_out # in bar # Note Cv_out in (m^3/s/bar)
            Rgas = 8.3145    # J/mol/K
            # Pressure profiles
            P_av = (P_out+P_feed)/2     # in bar
            DPDz_term1 = -1.75*P_av*1E5/R_gas/T_feed*(1-epsi)/epsi*u_feed**2
            DPDz_term2 = -150*mu_av/dp**2*(1-epsi)**2/epsi**2*u_feed
            DPDz = DPDz_term1 + DPDz_term2          # in (Pa/m)
            P_0 = P_L + DPDz*1E-5                   # in bar
            P_in = P_0 + self._Q_in/self._Cv_in     # in bar # in bar # Note Cv_out in (m^3/s/bar)
            # P profile once again
            P_av = (P_out+P_in)/2
            DPDz_term1 = -1.75*P_av*1E5/R_gas/T_feed*(1-epsi)/epsi*u_feed**2
            DPDz_term2 = -150*mu_av/dp**2*(1-epsi)**2/epsi**2*u_feed
            DPDz = DPDz_term1 + DPDz_term2 # in (Pa/m)
            P_0 = P_L + DPDz*1E-5
            P_in = P_0 + self._Q_in/self._Cv_in
            P = DPDz*1E-5*z + P_0   # bar
            # C profile
            C = P*1E5/R_gas/T_feed   # mol/m^3
            # Mass transfer ...
            k_mtc = self._k_mtc
            D_AB = self._D_disp
            a_surf = self._a_surf

            def massbal_linvel(y,t):
                y_comp = []
                P_part = []
                dy_comp = []
                ddy_comp = []
                for ii in range(n_comp-1):
                    y_tmp = np.array(y[ii*N:(ii+1)*N])
                    y_tmp[y_tmp<1E-6] = 0
                    dy_tmp = self._d@y_tmp
                    ddy_tmp =self._dd@y_tmp
                    P_part_tmp = y_tmp*P
                    #y_tmp[y_tmp<1E-6] = 0
                    y_comp.append(y_tmp)
                    P_part.append(P_part_tmp)
                    dy_comp.append(dy_tmp)
                    ddy_comp.append(ddy_tmp)
                y_rest = 1-np.sum(y_comp, 0)
                P_part.append(P*y_rest)
                #print('Shape of P_part[0]',P_part[0].shape)
                q = []
                for ii in range(n_comp-1, 2*n_comp-1):
                    q.append(y[ii*N:(ii+1)*N])
                # Solid uptake component
                dqdt = []
                qstar = self._iso(P_part,T_feed)
                for ii in range(n_comp):
                    dqdt_tmp = k_mtc[ii]*a_surf*(qstar[ii] - q[ii])
                    dqdt.append(dqdt_tmp)
                # Solid uptake total
                Sig_dqdt = np.sum(dqdt, 0)
                #print(Sig_dqdt)
                dy_compdt = []
                for ii in range(n_comp-1):
                    term1 = -u_feed*dy_comp[ii]
                    term2 = D_AB[ii]*0 # How to calculate d(yC)dz
                    term3 = -rho_s*(1-epsi)/epsi*dqdt[ii]
                    term4 = 0
                    term5 = +y_comp[ii]*rho_s*Sig_dqdt*(1-epsi)/epsi
                    
                    #term1[0] = 0
                    term1[0] = u_feed*self._d[1,1]*(y_feed[ii]-y_comp[ii][0])
                    #term5[0] = 0
                    #term1[-1] = -u_feed*d[1,1]*C[-1]*(y_comp[ii][-2]-y_comp[ii][-1])
                    #term3[-1] = 0
                    #term5[-1] = 0

                    #dydt_tmp = term1+(term2 + term3+ term4 + term5)/C
                    dydt_tmp = term1 + term3/C + term5/C
                    #dydt_tmp[0] = 0
                    #dydt_tmp[N-1] = 0 
                    #ind_both = ((y_comp[ii]<1E-6) + (dydt_tmp < 0))>1.5
                    #dydt_tmp[ind_both] = 0 
                    dy_compdt.append(dydt_tmp)
                dydt = []
                for yy in dy_compdt:
                    dydt.append(yy)
                    #print(yy.shape)
                for qq in dqdt:
                    dydt.append(qq)
                    #print(qq.shape)
                dydt_arr = np.concatenate(dydt)
                
                return dydt_arr
            
            # Initial value again
            y0 = np.zeros(N*(n_comp*2-1))
            for ii in range(n_comp-1):
                y0[ii*N : (ii+1)*N] = self._y_init[ii]
                y0[(n_comp-1)*N+ii*N:(n_comp-1)*N+(ii+1)*N] = self._q_init[ii]
            y0[2*(n_comp-1)*N:2*(n_comp-1)*N+N] = self._q_init[-1]
            
            # Backward flow case
            if self._required['Flow direction'] == 'Backward':
                y0_tmp = []
                for ii in range(2*n_comp-1):
                    mat_y0_pre_tmp = y0[ii*N:(ii+1)*N]
                    mat_y0_rev_tmp = self._A_flip@mat_y0_pre_tmp
                    y0_tmp.append(mat_y0_rev_tmp)
                y0_rev = np.concatenate(y0_tmp)
                y0 = y0_rev
                
            # Solve ode
            y_result = odeint(massbal_linvel, y0, t_dom, rtol=1e-6, atol=1e-9)
            
            # BACKWARD FLOW !! #
            # Flipped one for backward flow
            if self._required['Flow direction'] == 'Backward':
                y_tmp = []
                q_tmp = []
                for ii in range(n_comp-1):
                    mat_y_tmp = y_result[:, ii*N : (ii+1)*N]
                    mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                    y_tmp.append(mat_y_tmp@self._A_flip)
                    q_tmp.append(mat_q_tmp@self._A_flip)
                mat_q_tmp = y_result[:, (2*n_comp-2)*N:(2*n_comp-1)*N]
                q_tmp.append(mat_q_tmp@self._A_flip)
                y_flip = np.concatenate(y_tmp+q_tmp, axis = 1)
                y_result = y_flip
            
            # Assign the results (packaging the results)
            C_tmp = []
            q_tmp = []
            y_tmp = []
            mat_y_rest = np.zeros([len(t_dom), N])
            for ii in range(n_comp-1):
                mat_y_tmp = y_result[:, ii*N:(ii+1)*N]
                mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                mat_C_tmp = mat_y_tmp@np.diag(C)
                
                C_tmp.append(mat_C_tmp)
                q_tmp.append(mat_q_tmp)
                y_tmp.append(mat_y_tmp)
                mat_y_rest = mat_y_rest + mat_y_tmp

            C_tmp.append((1- mat_y_rest)@np.diag(C))
            q_tmp.append(y_result[:, (2*n_comp-2)*N: (2*n_comp-1)*N])
            
            self._y_fra = y_tmp

            Tg_tmp = T_feed*np.ones([len(t_dom), N])
            Ts_tmp = T_feed*np.ones([len(t_dom), N])
            y_result = np.concatenate(C_tmp + q_tmp + [Tg_tmp, Ts_tmp], axis=1)
            self._y = y_result
            self._t = t_dom

            toc = time.time()/60 - tic
            self._CPU_min = toc
            self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))
            
            return y_result, self._z, t_dom

        # Pressurization (open inlet only)
        elif (self._Cv_in > 1E-10) and (self._Cv_out < 1E-10): 
            # Initial Pressure and initial overal concentration
            Rgas = 8.3145   # J/mol/K
            P_av0 = np.mean(self._P_init)
            C_av0 = P_av0*1E5/R_gas/T_feed   # mol/m^3
            # Mass transfer
            k_mtc = self._k_mtc
            D_AB = self._D_disp
            a_surf = self._a_surf

            def massbal_linvel(y,t):
                y_comp = []
                P_part = []
                dy_comp = []
                ddy_comp = []
                C = y[(2*n_comp-1)*N]
                P_av = C*Rgas*T_feed/1E5   # mol/m^3
                Q_feed = self._Cv_in*(P_feed - P_av)
                u_max = Q_feed/self._A/epsi
                u_vel = u_max*(1-z/L)
                if u_max > 0:
                    F_feed = Q_feed*(P_feed/Rgas/T_feed)*1E5
                else:
                    F_feed = Q_feed*(P_av/Rgas/T_feed)*1E5
                
                #Q_feed = P_av0*
                for ii in range(n_comp-1):
                    y_tmp = np.array(y[ii*N:(ii+1)*N])
                    y_tmp[y_tmp<1E-6] = 0
                    dy_tmp = self._d@y_tmp
                    ddy_tmp = self._dd@y_tmp
                    P_part_tmp = y_tmp*P_av
                    #y_tmp[y_tmp<1E-6] = 0
                    y_comp.append(y_tmp)
                    P_part.append(P_part_tmp)
                    dy_comp.append(dy_tmp)
                    ddy_comp.append(ddy_tmp)

                y_rest = 1-np.sum(y_comp, 0)
                P_part.append(P_av*y_rest)
                #print('Shape of P_part[0]',P_part[0].shape)
                q = []
                for ii in range(n_comp-1, 2*n_comp-1):
                    q.append(y[ii*N:(ii+1)*N])
                # Solid uptake component
                dqdt = []
                qstar = self._iso(P_part,T_feed)
                for ii in range(n_comp):
                    dqdt_tmp = k_mtc[ii]*a_surf*(qstar[ii] - q[ii])
                    dqdt.append(dqdt_tmp)
                # Solid uptake total
                Sig_dqdt = np.sum(dqdt, 0)
                dqdt_av = np.sum(Sig_dqdt)/len(Sig_dqdt)
                
                # dCdt: 1st derv term of average overall gas concentration 
                dCdt = 1/epsi/V_tot*F_feed -(1-epsi)/epsi*rho_s*dqdt_av
                dy_compdt = []
                for ii in range(n_comp-1):
                    term1 = -u_vel*dy_comp[ii]
                    term2 = D_AB[ii]*0 # How to calculate d(yC)dz
                    term3 = -rho_s*(1-epsi)/epsi*dqdt[ii]
                    term4 = 0
                    term5 = +y_comp[ii]*rho_s*Sig_dqdt*(1-epsi)/epsi
                    
                    #term1[0] = 0
                    term1[0] = u_vel[0]*self._d[1,1]*(y_feed[ii]-y_comp[ii][0])
                    #term5[0] = 0
                    #term1[-1] = -u_feed*d[1,1]*C[-1]*(y_comp[ii][-2]-y_comp[ii][-1])
                    #term3[-1] = 0
                    #term5[-1] = 0

                    #dydt_tmp = term1+(term2 + term3+ term4 + term5)/C
                    dydt_tmp = term1 + term3/C + term5/C
                    #dydt_tmp[0] = 0
                    #dydt_tmp[N-1] = 0 
                    #ind_both = ((y_comp[ii]<1E-6) + (dydt_tmp < 0))>1.5
                    #dydt_tmp[ind_both] = 0 
                    dy_compdt.append(dydt_tmp) # dy_i/dt i = 1,2, ...

                dydt = []
                for yy in dy_compdt:
                    dydt.append(yy)
                    #print(yy.shape)
                for qq in dqdt:
                    dydt.append(qq)
                    #print(qq.shape)
                dydt.append([dCdt])
                dydt_arr = np.concatenate(dydt)
                
                return dydt_arr
            
            # Initial value again
            y0 = np.zeros(N*(n_comp*2-1)+1)
            for ii in range(n_comp-1):
                y0[ii*N : (ii+1)*N] = self._y_init[ii]
                y0[(n_comp-1+ii)*N : (n_comp+ii)*N]= self._q_init[ii]
            y0[2*(n_comp-1)*N : 2*(n_comp-1)*N+N] = self._q_init[-1]
            y0[(2*n_comp-1)*N] = C_av0

            # BACKWARD FLOW !! Flip initial conditions #
            if self._required['Flow direction'] == 'Backward':
                y0_tmp = []
                for ii in range(2*n_comp-1):
                    mat_y0_pre_tmp = y0[ii*N:(ii+1)*N]
                    mat_y0_rev_tmp = self._A_flip@mat_y0_pre_tmp
                    y0_tmp.append(mat_y0_rev_tmp)
                y0_tmp.append([y0[-1]])
                y0_rev = np.concatenate(y0_tmp)
                y0 = y0_rev
                
            # Solve ode
            y_result = odeint(massbal_linvel, y0, t_dom, rtol=1e-6, atol=1e-9)

            # BACKWARD FLOW !! Flip the results#
            if self._required['Flow direction'] == 'Backward':
                y_tmp = []
                q_tmp = []
                for ii in range(n_comp-1):
                    mat_y_tmp = y_result[:, ii*N : (ii+1)*N]
                    mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                    y_tmp.append(mat_y_tmp@self._A_flip)
                    q_tmp.append(mat_q_tmp@self._A_flip)
                mat_q_tmp = y_result[:, (2*n_comp-2)*N:(2*n_comp-1)*N]
                q_tmp.append(mat_q_tmp@self._A_flip)
                y_flip = np.concatenate(y_tmp+q_tmp + [y_result[:,-1:]], axis = 1)
                y_result = y_flip

            C_t_dom = y_result[:, -1]
            C_tmp = []
            q_tmp = []
            y_tmp = []
            mat_y_rest = np.zeros([len(t_dom), N])
            for ii in range(n_comp-1):
                mat_y_tmp = y_result[:, ii*N:(ii+1)*N]
                mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                mat_C_tmp = np.diag(C_t_dom)@mat_y_tmp
                
                C_tmp.append(mat_C_tmp)
                q_tmp.append(mat_q_tmp)
                y_tmp.append(mat_y_tmp)
                mat_y_rest = mat_y_rest + mat_y_tmp

            C_tmp.append(np.diag(C_t_dom)@(1- mat_y_rest))
            q_tmp.append(y_result[:, (2*n_comp-2)*N: (2*n_comp-1)*N])

            self._y_fra = y_tmp

            Tg_tmp = T_feed*np.ones([len(t_dom), N])
            Ts_tmp = T_feed*np.ones([len(t_dom), N])
            y_result = np.concatenate(C_tmp + q_tmp + [Tg_tmp, Ts_tmp], axis=1)
            self._y = y_result
            self._t = t_dom

            toc = time.time()/60 - tic
            self._CPU_min = toc
            self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))

            return y_result, self._z, t_dom
        
        # Depressurization (=Blowdown) (Open outlet only)
        elif (self._Cv_in < 1E-10) and (self._Cv_out > 1E-10): 
            # Initial Pressure and initial overal concentration
            Rgas = 8.3145   # J/mol/K
            P_av0 = np.mean(self._P_init)
            C_av0 = P_av0*1E5/R_gas/T_feed   # mol/m^3
            # Mass transfer
            k_mtc = self._k_mtc
            D_AB = self._D_disp
            a_surf = self._a_surf

            def massbal_linvel(y,t):
                y_comp = []
                P_part = []
                dy_comp = []
                ddy_comp = []
                C = y[(2*n_comp-1)*N]
                P_av = C*Rgas*T_feed/1E5   # mol/m^3
                Q_feed = self._Cv_out*(P_av - P_out)
                u_max = Q_feed/self._A/epsi
                u_vel = u_max*z/L
                if u_max > 0:
                    F_feed = Q_feed*(P_av/Rgas/T_feed)*1E5
                else:
                    F_feed = Q_feed*(P_out/Rgas/T_feed)*1E5
                
                #Q_feed = P_av0*
                for ii in range(n_comp-1):
                    y_tmp = np.array(y[ii*N:(ii+1)*N])
                    y_tmp[y_tmp<1E-6] = 0
                    dy_tmp = self._d@y_tmp
                    ddy_tmp = self._dd@y_tmp
                    P_part_tmp = y_tmp*P_av
                    #y_tmp[y_tmp<1E-6] = 0
                    y_comp.append(y_tmp)
                    P_part.append(P_part_tmp)
                    dy_comp.append(dy_tmp)
                    ddy_comp.append(ddy_tmp)

                y_rest = 1-np.sum(y_comp, 0)
                P_part.append(P_av*y_rest)
                q = []
                for ii in range(n_comp-1, 2*n_comp-1):
                    q.append(y[ii*N:(ii+1)*N])
                # Solid uptake component
                dqdt = []
                qstar = self._iso(P_part,T_feed)
                for ii in range(n_comp):
                    dqdt_tmp = k_mtc[ii]*a_surf*(qstar[ii] - q[ii])
                    dqdt.append(dqdt_tmp)
                # Solid uptake total
                Sig_dqdt = np.sum(dqdt, 0)
                dqdt_av = np.sum(Sig_dqdt)/len(Sig_dqdt)
                
                # dCdt: 1st derv term of average overall gas concentration 
                dCdt = -1/epsi/V_tot*F_feed -(1-epsi)/epsi*rho_s*dqdt_av
                dy_compdt = []
                for ii in range(n_comp-1):
                    term1 = -u_vel*dy_comp[ii]
                    term2 = D_AB[ii]*0 # How to calculate d(yC)dz
                    term3 = -rho_s*(1-epsi)/epsi*dqdt[ii]
                    term4 = 0
                    term5 = +y_comp[ii]*rho_s*Sig_dqdt*(1-epsi)/epsi
                    
                    term1[0] = 0
                    # Because u_vel[0] = 0 ... 
                    # [???] term1[0] = u_vel[0]*self._d[1,1]*(y_feed[ii]-y_comp[ii][0])

                    #term5[0] = 0
                    #term1[-1] = -u_feed*d[1,1]*C[-1]*(y_comp[ii][-2]-y_comp[ii][-1])
                    #term3[-1] = 0
                    #term5[-1] = 0

                    #dydt_tmp = term1+(term2 + term3+ term4 + term5)/C
                    dydt_tmp = term1 + term3/C + term5/C
                    #dydt_tmp[0] = 0
                    #dydt_tmp[N-1] = 0 
                    #ind_both = ((y_comp[ii]<1E-6) + (dydt_tmp < 0))>1.5
                    #dydt_tmp[ind_both] = 0 
                    dy_compdt.append(dydt_tmp) # dy_i/dt i = 1,2, ...

                dydt = []
                for yy in dy_compdt:
                    dydt.append(yy)
                    #print(yy.shape)
                for qq in dqdt:
                    dydt.append(qq)
                    #print(qq.shape)
                dydt.append([dCdt])
                dydt_arr = np.concatenate(dydt)
                
                return dydt_arr
            
            # Initial value again
            y0 = np.zeros(N*(n_comp*2-1)+1)
            for ii in range(n_comp-1):
                y0[ii*N : (ii+1)*N] = self._y_init[ii]
                y0[(n_comp-1+ii)*N : (n_comp+ii)*N]= self._q_init[ii]
            y0[2*(n_comp-1)*N : 2*(n_comp-1)*N+N] = self._q_init[-1]
            y0[(2*n_comp-1)*N] = C_av0
            
            # BACKWARD FLOW !! Flip initial conditions #
            if self._required['Flow direction'] == 'Backward':
                y0_tmp = []
                for ii in range(2*n_comp-1):
                    mat_y0_pre_tmp = y0[ii*N:(ii+1)*N]
                    mat_y0_rev_tmp = self._A_flip@mat_y0_pre_tmp
                    y0_tmp.append(mat_y0_rev_tmp)
                y0_tmp.append([y0[-1]])
                y0_rev = np.concatenate(y0_tmp)
                y0 = y0_rev
                
            # Solve ode
            y_result = odeint(massbal_linvel, y0, t_dom, rtol=1e-6, atol=1e-9)

            # BACKWARD FLOW !! Flip the results#
            if self._required['Flow direction'] == 'Backward':
                y_tmp = []
                q_tmp = []
                for ii in range(n_comp-1):
                    mat_y_tmp = y_result[:, ii*N : (ii+1)*N]
                    mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                    y_tmp.append(mat_y_tmp@self._A_flip)
                    q_tmp.append(mat_q_tmp@self._A_flip)
                mat_q_tmp = y_result[:, (2*n_comp-2)*N:(2*n_comp-1)*N]
                q_tmp.append(mat_q_tmp@self._A_flip)
                y_flip = np.concatenate(y_tmp+q_tmp + [y_result[:,-1:]], axis = 1)
                y_result = y_flip

            C_t_dom = y_result[:, -1]
            C_tmp = []
            q_tmp = []
            y_tmp = []
            mat_y_rest = np.zeros([len(t_dom), N])
            for ii in range(n_comp-1):
                mat_y_tmp = y_result[:, ii*N:(ii+1)*N]
                mat_q_tmp = y_result[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
                mat_C_tmp = np.diag(C_t_dom)@mat_y_tmp
                
                C_tmp.append(mat_C_tmp)
                q_tmp.append(mat_q_tmp)
                y_tmp.append(mat_y_tmp)
                mat_y_rest = mat_y_rest + mat_y_tmp

            C_tmp.append(np.diag(C_t_dom)@(1- mat_y_rest))
            q_tmp.append(y_result[:, (2*n_comp-2)*N: (2*n_comp-1)*N])

            self._y_fra = y_tmp

            Tg_tmp = T_feed*np.ones([len(t_dom), N])
            Ts_tmp = T_feed*np.ones([len(t_dom), N])
            y_result = np.concatenate(C_tmp + q_tmp + [Tg_tmp, Ts_tmp], axis=1)
            self._y = y_result
            self._t = t_dom

            toc = time.time()/60 - tic
            self._CPU_min = toc
            self._Tg_res = y_result[:,n_comp*2*N : n_comp*2*N+N]
            if CPUtime_print:
                print('Simulation of this step is completed.')
                print('This took {0:9.3f} mins to run. \n'.format(toc))

            return y_result, self._z, t_dom
            
            
        
        

        



#### P equalization ??? ####
def step_P_eq_alt1(column1, column2, t_max,
n_sec=5, Cv_btw=0.1, valve_select = [1,1], CPUtime_print = False):
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
    #print(C_sta1)

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
    y_res = odeint(massmomeenbal_eq_gasonly, y0_tot_gas,t_dom_tenth, rtol=1e-6, atol=1e-9)

    # Update concentration
    y0_tot[:N1*n_comp] = y_res[-1,:N1*n_comp]
    y0_tot[n_var_tot1:n_var_tot1+N2*n_comp] = y_res[-1,N1*n_comp:]
    # Update temperature
    y0_tot[n_var_tot1-2*N1:n_var_tot1-1*N1] = T_mean*np.ones(N1)
    y0_tot[-2*N2:-1*N2] = T_mean*np.ones(N2)
    
    #RUN2
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom, rtol=1e-6, atol=1e-9)
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
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom, rtol=1e-6, atol=1e-9)
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
    y_res = odeint(massmomeenbal_eq, y0_tot,t_dom, rtol=1e-6, atol=1e-9)
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
    N = 101
    
    D = 0.2
    A_cros = D**2/4*np.pi
    epsi = 0.4      # m^3/m^3
    rho_s = 1000    # kg/m^3
    dp = 0.02       # m (particle diameter)
    mu = [1.81E-5, 1.81E-5, 1.81E-5] # Pa sec (visocisty of gas)
    #mu_av = 1.81E-5
    n_comp = 3
    
    # Column Geometry
    L = 1
    D = 0.2
    N = 101
    A_cros = D**2/4*np.pi
    epsi = 0.4      # m^3/m^3
    # Define a column    
    n_comp = 3
    c1 = column(L, A_cros,n_comp, N, E_balance=False)

    # Adsorbent info
    def isomix(P_in,T):
        pp = []
        for ii in range(len(P_in)):
            p_tmp = P_in[ii][:]
            #ind_tmp = P_in[ii] < 1E-4
            #p_tmp[ind_tmp] = 0
            pp.append(p_tmp)
        q1 = 1*pp[0]*0.1/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
        q2 = 3*pp[1]*0.3/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
        q3 = 4*pp[2]*0.4/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
        return [q1, q2, q3]
    c1.adsorbent_info(isomix, epsi, dp, rho_s,)
    
    # Gas properties
    Mw = [28, 32, 44]
    c1.gas_prop_info(Mw,mu)

    # Mass transfer
    k_MTC = [2.5, 2.5, 20.5]
    D_disp = [1E-7, 1E-7, 1E-7] 
    a_surf = 1
    c1.mass_trans_info(k_MTC, a_surf, D_disp)

    # Boundary conditions
    P_out = 1.2
    P_in = 1.7 # Ignore this
    T_in = 300
    y_in = [0.45, 0.3, 0.25]
    Cv_in = 5E-3        # m^3/sec/bar
    Cv_out = 5E-3       # m^3/sec/bar
    u_feed = 0.1            # m/s
    Q_in = u_feed*A_cros*epsi  # volumetric flowrate
    c1.boundaryC_info(P_out, P_in, T_in, y_in, Cv_in,Cv_out,Q_in, 
                      assigned_v_option = True, foward_flow_direction=True)
    # Initial conditions
    P_init = 2*np.ones([N,])
    
    y_init = [0.75*np.ones([N,]),
              0.25*np.ones([N,]),
              0.00*np.ones([N,])]
    P_part = [0.75*np.ones([N,])*P_init,
              0.25*np.ones([N,])*P_init,
              0.00*np.ones([N,])*P_init]
    Tg_init = 300*np.ones([N,])
    Ts_init = 300*np.ones([N,])
    q1,q2,q3 = isomix(P_part, Tg_init)
    q_init = [q1,q2,q3]

    c1.initialC_info(P_init, Tg_init, Ts_init, y_init, q_init )
    print(c1)
    
    y_res, z_res, t_res = c1.run_ma_linvel(100,10)
    c1.Graph(20, 0)

    '''
    N = 11
    A_cros = 0.031416
    L = 1
    c1 = column(L,A_cros, n_component = 2,N_node = N)
    '''
    '''
    dP = np.linspace(-100, 100)
    M_m_test  = [0.044, 0.028]      ## molar mass    (kg/mol)
    mu_test = [1.47E-5, 1.74E-5]    ## gas viscosity (Pa sec)
    D_particle_dia = 0.01   # particle diameter (m)
    epsi_test = 0.4         # macroscopic void fraction (m^3/m^3)
    v_test = Ergun_test(dP,M_m_test[0],mu_test[0],D_particle_dia,epsi_test)
    plt.plot(dP,v_test)
    plt.grid()
    plt.show()
    '''

    '''
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
    rho_s_test = 1100       # solid density (kg/m^3)
    c1.adsorbent_info(iso_fn_test,epsi_test,D_particle_dia, rho_s_test)
 
    M_m_test  = [0.044, 0.028]      ## molar mass    (kg/mol)
    mu_test = [1.47E-5, 1.74E-5]    ## gas viscosity (Pa sec)
    c1.gas_prop_info(M_m_test, mu_test)
 
    ## Mass transfer coefficients
    D_dis_test = [1E-6, 1E-6]   # m^2/sec
    k_MTC = [0.0002, 0.0002]    # m/sec
    a_surf = 200                # m^2/m^3
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
    '''

    '''
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
    '''
    
# %%

# %%
