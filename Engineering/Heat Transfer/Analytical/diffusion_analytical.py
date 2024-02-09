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
from NewtonRaphson import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class CylinderNewton:
    def __init__(self,Biot,n=30):
        self.biot = Biot
        self.eigen = []
        self._calculate_eigenvalues(n)
        self._root_refinement()

    def transcendentalFunction(self,x):
        return x*jv(1,x)-self.biot*jv(0,x)

    def derivative_transcendentalFunction(self,x):
        return (1.0 + self.biot)*jv(1,x) + x/2.0*(jv(0,x) - jv(2,x))

    def _calculate_eigenvalues(self,n):
        x0 = 0.0
        dx = 1e-2
        fx = 1.0
        fdx = 1.0
        x = x0
        xdx = x0 + dx
        has_root = False
        for i in xrange(n):
            while(not has_root):
                x = xdx
                xdx = x + dx
                fx = self.transcendentalFunction(x)
                fdx = self.transcendentalFunction(xdx)
                if(fx*fdx <= 0.0):
                    has_root = True
            root = (x + xdx)/2.0
            self.eigen.append(root)
            has_root = False
        
    def _root_refinement(self,eps=1.0e-8):
        i = 0
        for root in self.eigen:
            tmp = NewtonRaphson(self.transcendentalFunction, self.derivative_transcendentalFunction, root, eps)
            self.eigen[i] = tmp
            i += 1
    def T(self,r,t):
        tmp = 0.0
        for root in self.eigen:
            tmp += jv(1,root)*jv(0,root*r)/root/(jv(1,root)*jv(1,root) + jv(0,root)*jv(0,root))*np.exp(-root**2*t)
        tmp *= 2.0
        return tmp

    def dTdt(self,r,t):
        tmp = 0.0
        for root in self.eigen:
            tmp -= root*jv(1,root)*jv(0,root*r)/(jv(1,root)**2 + jv(0,root)**2)*np.exp(-root**2*t)
        tmp *= 2.0
        return tmp

    def t(self,rdim,Tdim):
        t0 = 0.0
        dt = 0.001
        time = t0
        timedt = t0 + dt
        
        f = lambda t: Tdim - self.T(rdim,t)
        has_root = False
        while(not has_root):
            ft = f(time)
            fdt = f(timedt)
            if(ft*fdt <= 0.0):
                has_root = True
            else:
                time = timedt
                timedt = time + dt
        root = (time + timedt)/2.0
        root = Bissection(f,time,timedt,epsilon=1e-7)
        return root


    def Surface_T(self,tmin,tmax,rmin,rmax,n_t=50,n_r=50):
        dt = (tmax - tmin)/n_t
        dr = (rmax - rmin)/n_r
        t,r = np.meshgrid(np.linspace(tmin, tmax, num=n_t),np.linspace(rmin,rmax,n_r), sparse=False,indexing='ij')
        Z = self.T(r,t)
        return t,r,Z

class Rectangular1DNewton:
    def __init__(self,Biot,n=20):
        self.biot = Biot
        self.eigen = []
        self._calculate_eigenvalues(n)
        self._root_refinement()

    def transcendentalFunction(self,x):
        return np.sin(x)*x - self.biot*np.cos(x)

    def derivative_transcendentalFunction(self,x):
        return np.cos(x)*x + np.sin(x) + self.biot*np.sin(x)

    def _calculate_eigenvalues(self,n):
        x0 = 0.0
        dx = 1e-3
        fx = 1.0
        fdx = 1.0
        x = x0
        xdx = x0 + dx
        has_root = False
        for i in xrange(n):
            while(not has_root):
                x = xdx
                xdx = x + dx
                fx = self.transcendentalFunction(x)
                fdx = self.transcendentalFunction(xdx)
                if(fx*fdx <= 0.0 and abs(fx - fdx) < 100.0):
                    has_root = True
            root = (x + xdx)/2.0
            
            self.eigen.append(root)
            has_root = False
        
    def _root_refinement(self,eps=1.0e-8):
        i = 0
        for root in self.eigen:
            tmp = NewtonRaphson(self.transcendentalFunction, self.derivative_transcendentalFunction, root, eps)
            self.eigen[i] = tmp
            i += 1
    def T(self,x,t):
        tmp = 0.0
        for root in self.eigen:
            tmp += np.sin(root)*np.cos(root*x)/(root + np.sin(root)*np.cos(root))*np.exp(-root**2*t)
        tmp *= 2.0
        return tmp

    def dTdt(self,x,t):
        tmp = 0.0
        for root in self.eigen:
            tmp -= root**2*np.sin(root)*np.cos(root*x)/(root + np.sin(root)*np.cos(root))*np.exp(-root**2*t)
        tmp *= 2.0
        return tmp


class Rectangular2DNewton:
    def __init__(self,Biot_x,Biot_y,n=50):
        self.biot_x = Biot_x
        self.Biot_y = Biot_y
        self.eigen_x = []
        self.eigen_y = []
        self._calculate_eigenvalues(n)
        self._root_refinement()

    def transcendentalFunction_x(self,x):
        return np.sin(x)*x - Bi*np.cos(x)

    def derivative_transcendentalFunction_x(self,x):
        return np.cos(x)*x + np.sin(x) + self.biot*Bi*np.sin(x)

    def transcendentalFunction_y(self,y):
        return np.sin(y)*x - self.biot_y*np.cos(y)

    def derivative_transcendentalFunction_y(self,y):
        return np.cos(y)*x + np.sin(y) + self.biot_y*np.sin(y)

    def _calculate_eigenvalues(self,n):
        x0 = 0.0
        dx = 1e-3
        fx = 1.0
        fdx = 1.0
        x = x0
        xdx = x0 + dx
        has_root = False
        for i in xrange(n):
            while(not has_root):
                x = xdx
                xdx = x + dx
                fx = self.transcendentalFunction_x(x)
                fdx = self.transcendentalFunction_x(xdx)
                if(fx*fdx <= 0.0 and abs(fx - fdx) < 100.0):
                    has_root = True
            root = (x + xdx)/2.0
            
            self.eigen_x.append(root)
            has_root = False
        x0 = 0.0
        dx = 1e-3
        fx = 1.0
        fdx = 1.0
        x = x0
        xdx = x0 + dx
        has_root = False
        for i in xrange(n):
            while(not has_root):
                x = xdx
                xdx = x + dx
                fx = self.transcendentalFunction_y(x)
                fdx = self.transcendentalFunction_y(xdx)
                if(fx*fdx <= 0.0 and abs(fx - fdx) < 100.0):
                    has_root = True
            root = (x + xdx)/2.0
            
            self.eigen_y.append(root)
            has_root = False
        
    def _root_refinement(self,eps=1.0e-8):
        i = 0
        for root in self.eigen_x:
            tmp = NewtonRaphson(self.transcendentalFunction_x, self.derivative_transcendentalFunction_x, root, eps)
            self.eigen_x[i] = tmp
            print tmp
            i += 1
            root = root/sqrt(2.0)

        i = 0
        for root in self.eigen_y:
            tmp = NewtonRaphson(self.transcendentalFunction_y, self.derivative_transcendentalFunction_y, root, eps)
            self.eigen_y[i] = tmp
            print tmp
            i += 1
            root = root/sqrt(2.0)

    def T(self,x,y,t):
        tmp = 0.0
        for root in self.eigen_x:
            for root in self.eigen_y:
                tmp += np.sin(root)*np.cos(root*x)/(root + np.sin(root)*np.cos(root))*np.exp(-root**2*t)
        for root in self.eigen_y:
            tmp += np.sin(root)*np.cos(root*y)/(root + np.sin(root)*np.cos(root))*np.exp(-root**2*t)
        tmp *= 2.0
        return tmp



            
