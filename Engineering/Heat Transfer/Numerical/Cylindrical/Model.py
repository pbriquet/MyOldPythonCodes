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
from scipy.optimize import curve_fit, fmin, minimize_scalar
from matplotlib import cm
from VF import *

class CylindricalModel:
    def __init__(self,R,T0,nr):
        self.n = nr
        self.T0 = T0
        self.R = R
        self.mesh = CylindricalVFMesh(R,nr)

    def setInitialConditions(self,T0):
        if(T0 is float):
            self.T0 = T0
            self.f = lambda x: T0
        else:
            self.T0 = -minimize_scalar(lambda x: -T0(x), bounds=[0,self.R], method='bounded').fun
            self.f = T0

        for vf in self.mesh.vfs:
            vf.T = self.f(vf.rc)

    def setBoundaryConditions(self,h,Tfar):
        self.h = h
        self.Tfar = Tfar

    def setRadiation(self,epsilon):
        self.stefan = 5.67036e-8

    def setMaterial(self,K,rho,Cp):
        self.K = K
        self.rho = rho
        self.Cp = Cp
        self.alpha = K/rho/Cp

    def Dimensionless(self):
        self.tau = self.R**2/self.alpha
        self.Biot = self.h*self.R/self.K
        self.dT = self.T0 - self.Tfar
        self.Tdim = lambda T: (T - self.Tfar)/self.dT
    
