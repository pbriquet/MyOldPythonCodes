import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from Stencil import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from HB_Data import *
from hardnessmodel import *
from scipy import optimize
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import cm
from hardnessmodel import *
from hardnessevolution import *


Tmin, Tmax = [430.0,760.0]
H0 = 500.0
H_inf = 200.0
Treach = np.linspace(580,660.0,8,endpoint=True)
T0 = 200.0
dTdt = 60.0 # C/h
T_iso = 630.0
tmax = 36.0
dtmax = 0.1

a0 = [5.13307046e-01,1.11841489e+03,6.28526467e+04,4.78132101e+03]

h = HardnessModel()
h.change_parameters(H0,H_inf,Tmin,Tmax)
h.functional(a0[0],a0[1],a0[2],a0[3])
(self,m,D0,QD,Qinf):
i = 0
for T in Treach:
    tmp = HardnessEvolution(h,tmax,dtmax)
    tmp.setInitialConditions(T0,T,dTdt)
    t,tau,alpha,Temp = tmp.calculate()
    H = map(h.H_from_tau, tau)
    H_iso = map(lambda x: h.H(x,T),t)    
    plt.plot(t,H,'C'+str(i),label='Rampa ' + "{0:.2f}".format(dTdt) + ' C/h T = ' + "{0:.2f}".format(T))
    plt.plot(t,H_iso,'C'+str(i)+'--',label='Isotermico T = ' + "{0:.2f}".format(T) + ' C')
    i+=1

Hmin = map(lambda x: 269.0, t)
Hmax = map(lambda x: 293.0, t)


#plt.plot(Treach,Hfar)
plt.plot(t,Hmin,"b:",label='H min')
plt.plot(t,Hmax,"r:",label='H max')
plt.xlabel('t (h)')
plt.ylabel('HB')
plt.legend()
plt.show()
