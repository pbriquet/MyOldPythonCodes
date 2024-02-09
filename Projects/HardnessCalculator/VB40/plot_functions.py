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

a0 = [5.19148706e-01,3.97545668e+02,5.50929328e+04,4.43595129e+03]
Tmin, Tmax = [420.0,751.0]
[5.13307046e-01,1.11841489e+03,6.28526467e+04,4.78132101e+03]
T = np.linspace(Tmin,Tmax,300)
h = HardnessModel()
h.functional(a0[0],a0[1],a0[2],a0[3])
theta = map(h.theta,T)
dT, dtheta = derivative_arrays(theta,T)
dT2, ddtheta = derivative_arrays(dtheta,dT)
plt.plot(dT2,ddtheta,'r')

plt.show()