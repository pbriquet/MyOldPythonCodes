import numpy as np

#sigma_fe = Intrisic Lattice Strength of Fe (Peierls-Nabarro)
##sigma_ss = Solid Solution Hardening of Steel (Lacy)
#sigma_P = Precipitation Hardness
#sigma_dis = Dislocation Strengthening
#sigma_sgb = Subgrain strengthening
def yield_stress(sigma_fe,sigma_ss,sigma_P,sigma_dis,sigma_sgb)
    return sigma_fe + sigma_ss + sigma_P + sigma_dis + sigma_sgb

yield_stress()
def yield_fe(T):
    return 78.0 - 0.023*(T + 273.15)

def yield_ss(X):
    K = {
        'C':11000
    }
    tmp = 0.0
    for k,v in X.items():
        if(k in K.keys()):
            tmp += v*K[k]
    return tmp

def yield_P(N_total,N_coh,N_incoh,sigma_chem,sigma_coh,sigma_mod,sigma_0):
    return np.power(N_coh/N_total,1/2)*np.sqrt(
        np.power(sigma_chem,2) + 
        np.power(sigma_coh,2) + 
        np.power(sigma_mod,2)
        ) + np.power(N_incoh/N_total,1/2)*sigma0

def yield_chem(M,b,lamb,Gamma,gamma):
    return 2.0*M/b/lamb/np.power(Gamma,1/2)*np.power(gamma*b,3/2)

def yield_coh(M,G,epsilon,N,b,r):
    return 8.4*M*G*np.power(epsilon,3/2)*np.power(N/b,1/2)*np.power(r,2)

def yield_mod(M,G,b,lamb,U_p,U_m):
    return M*G*b/lamb*np.power(1 - (U_p/U_m),3/4)

def yield_0(C,M,G,b,lamb,r0,ri)
    return C*M*G*b/lamb*np.log(r0/ri)

def yield_dis(alpha,G,b,rho_dis):
    return alpha*G*b*np.sqrt(rho_dis)

def alpha_func(T):
    return 0.4 - 4.42e-4*(T + 273.15)

def G_func(T):
    return -0.039431*(T + 273.15) + 98.46456

def rho_dis_0(Ms):
    return np.power(10.0,9.2840 + 6880.73/(Ms + 273.15) - 1780360.0/np.power(Ms + 273.15,2))

def drho_dis_dt(C_dis,rho_dis,Q_v,R,T):
    return C_dis*rho_dis*np.exp(-Q_v/R/(T + 273.15))

def yield_sgb(w):
    return 115.0/2.0/w 

def dw_dt(C_w,w,Q_m,R,T):
    return C_w*w*np.exp(-Q_m/R/(T+273.15))
