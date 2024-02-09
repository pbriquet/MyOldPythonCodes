from math import *
import matplotlib.pyplot as plt
import numpy as np

def Nu(Re,Pr):
    return 2.0*0.332*np.sqrt(Re)*np.power(Pr,(1.0/3.0))

mu = 1.81e-5    # Pa.s
k = 0.0262  # W/m/K
Cp = 718    # J/kg/K
rho = 1.228 # kg/m^3
D = 0.02
Pr = rho*Cp*mu/k # 0.496022
emissivity1 = 0.7
stefan_boltzmann = 5.670373e-8

v = 10.0
v = np.linspace(1e-3,30.0,num=100)
h_rad = stefan_boltzmann*emissivity1*((1100.0 + 273.15)**3 + (25.0 + 273.15)**3)
print(h_rad)
Re = rho*v*D/mu
_Nu = Nu(Re,Pr)
h = _Nu*k/D
plt.plot(v,h)
plt.show()
