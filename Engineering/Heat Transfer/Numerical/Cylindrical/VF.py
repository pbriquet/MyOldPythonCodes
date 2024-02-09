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

class CylindricalVF:
    def __init__(self,r,re,rw):
        self.re = re
        self.rw = rw
        self.rc = r
        self.r = [self.rw,self.rc,self.re]
        self.ri = [self.rw,self.re]
        self.T = 0.0
        self.Tdt = 0.0
    def __str__(self):
        return "rw = " + str(self.rw) + '\trc = ' + str(self.rc) + '\tre = ' + str(self.re) + '\tT = ' + str(self.T)

class CylindricalVFMesh:
    def __init__(self,R,nr):
        self.vfs = []
        self.dr = R/nr
        for i in range(nr):
            self.vfs.append(CylindricalVF((i+0.5)*self.dr,(i + 1)*self.dr, i*self.dr))

    def __str__(self):
        tmp = ''
        for i in self.vfs:
            tmp += str(i) + '\n'
        return tmp