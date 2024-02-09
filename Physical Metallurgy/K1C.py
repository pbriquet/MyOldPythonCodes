import numpy as np
from scipy import special

def K1C(sigma_y,phi,N,kappa):
    total = 0.0
    m = int((N + 1)/2)
    for k in range(0,m+1):
        total += special.binom(N,2*k-1)/np.power(kappa,N-(2*k-1))
    return sigma_y*np.sqrt(2.0*np.sqrt(2.0)*np.pi*np.power(N*total,N-1))

def K1Clim(sigma_u,phi,kappa):
    return sigma_u*np.sqrt(2.0*np.sqrt(2.0)*np.pi*phi/kappa)

sigma = 1e6
phi = 1.0
N = 2
kappa = 1e-6

x = K1C(sigma,phi,N,kappa)
y = K1Clim(sigma,phi,kappa)
print("{:.6e} {:.6e}".format(x,y))