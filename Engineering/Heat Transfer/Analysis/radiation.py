import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from collections import Counter
from scipy.special import jv,cotdg
from matplotlib.ticker import LinearLocator, FormatStrFormatter

stefanboltzmann = 5.67036e-8

def hr(T,Tfar,epsilon):
    return stefanboltzmann*epsilon*(T**3 + T**2*Tfar + T*Tfar**2 + Tfar**3)

epsilon = 1.0
Tfar = 1250.0 + 273.15
T = np.arange(1030.0 + 273.15,1250.0 + 273.15,1.0)
Y = hr(T,Tfar,epsilon)
plt.plot(T,Y)
plt.xlabel('T(K)')
plt.ylabel('hr')
plt.show()