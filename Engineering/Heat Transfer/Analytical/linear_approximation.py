import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from scipy.special import jv,cotdg
from enum import IntEnum
from enum import Enum
from diffusion_analytical import *

def dimensionless_time(Tdim,Bi):
    a0 = 0.00308125739688
    a1 = -0.496298387751
    b0 = 0.106238066206
    b1 = -0.148468728665
    return (a1*log(Tdim) + a0)/Bi + (b0 + b1*log(Tdim))

alpha = 4.0e-6
R = 0.1
h = 50.0
hrad = 650.0
htotal = h + hrad
K = 15.0
Biot = htotal*R/K
dT = 5.0
Tfar = 1100.0
T0 = 800.0
Tdim = ((Tfar - dT) - Tfar)/(T0 - Tfar)
print Tdim, Biot
print dimensionless_time(Tdim,Biot)*R**2/alpha/3600.0

c = CylinderNewton(Biot)
t = c.t(0.0,Tdim)*R**2/alpha/3600.0
tdim = c.t(0.0,Tdim)
T = c.T(0.0,tdim)
print t
print T