import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import jv,cotdg
from scipy import optimize as opt
import os

class RootFinder:
    # Search for roots in intervals
    @staticmethod
    def Bolzano(f,x0,xn,dx):    
        n = int((xn-x0)/dx) + 1
        roots = []
        for i in range(n):
            x = x0 + i*dx
            xdx = x + dx
            fx = f(x)
            fxdx = f(xdx)
            if(fx*fxdx <= 0.0):
                roots.append((x+xdx)/2.0)
        return np.array(roots)
    @staticmethod
    def searchInterval(f,x0):
        xn = x0
        xant = xn
        while(f(xant)*f(xn) > 0.0):
            xant = xn
            if(f(xn) < 0.0):
                xn = xn/2.0
            else:
                xn = xn*2.0
        return min([xant,xn]),max([xant,xn])
    @staticmethod
    def BolzanoRootCounter(f,x0,dx,n_roots,x_lim=1e5):    
        roots = []
        has_root = False
        x = x0
        xdx = x0
        for i in range(n_roots):
            while(not has_root and x < x_lim):
                x = xdx
                xdx = x + dx
                fx = f(x)
                fxdx = f(xdx)
                if(fx*fxdx <= 0.0):
                    has_root = True
            if(x < x_lim):
                roots.append((x+xdx)/2.0)
                has_root = False
            else:
                return None
        return np.array(roots)
    @staticmethod
    def BolzandoRootCounterIntervals(f,x0,dx,n_roots,x_lim=1e5):
        a = []
        b = []
        has_root = False
        x = x0
        xdx = x0
        for i in range(n_roots):
            while(not has_root and x < x_lim):
                x = xdx
                xdx = x + dx
                fx = f(x)
                fxdx = f(xdx)
                if(fx*fxdx <= 0.0):
                    has_root = True
            if(x < x_lim):
                a.append(x)
                b.append(xdx)
                has_root = False
            else:
                return None
        return np.array(a), np.array(b)


    @staticmethod
    def NewtonRaphson(f,dfdx,x0,error_max=1e-5):    
        error = 1.0
        n_attempts = 0
        n_attemps_max = 1000
        x = x0
        xant = x
        while(error > error_max and n_attempts < n_attemps_max):
            xant = x
            x = x - f(x)/dfdx(x)
            error = abs((x-xant)/xant)
            n_attempts += 1
        return x

class CylinderAnalytical1D:
    @staticmethod
    def transcendental(biot,x):
        return x*jv(1,x) - biot*jv(0,x)
    
    def __init__(self,BiotR,n_roots=30):
        self.biot = BiotR
        print("biot = " + str(BiotR))
        raw_roots = RootFinder.BolzanoRootCounter(self.f,0.0,1e-2,n_roots)
        self.eigenvalues = np.array([opt.newton(self.f,i) for i in raw_roots])
    def f(self,x):
        return x*jv(1,x) - self.biot*jv(0,x)
    def dfdx(self,x):
        return (1.0 + self.biot)*jv(1,x) + x/2.0*(jv(0,x) - jv(2,x))
    def _Ak(self,root,r,t):
        return jv(1,root)*jv(0,np.dot(root,r))/root/(jv(1,root)*jv(1,root) + jv(0,root)*jv(0,root))*np.exp(-np.dot(t,root**2))
    def T(self,r,t):
        return 2.0*np.sum(self._Ak(self.eigenvalues,r,t))
    def t(self,r,T):
        func = lambda t: self.T(r,t) - T
        x0 = 0.0
        if(self.biot < 0.1):
            x0 = -np.log(T)/self.biot/2.0
        else:
            x0 = 1.0
        a,b = RootFinder.searchInterval(func,x0)
        xopt = opt.bisect(func,a,b)
        return opt.newton(func,xopt)


class CylinderAnalyticalDirichlet1D:
    def __init__(self,n_roots=30):
        raw_roots = RootFinder.BolzanoRootCounter(self.f,0.0,1e-2,n_roots)
        self.eigenvalues = np.array([opt.newton(self.f,i) for i in raw_roots])
    def f(self,x):
        return jv(0,x)
    def dfdx(self,x):
        return -jv(1,x)
    def _Ak(self,root,r,t):
        return jv(1,root)*jv(0,np.dot(root,r))/root/(jv(1,root)*jv(1,root) + jv(0,root)*jv(0,root))*np.exp(-np.dot(t,root**2))
    def T(self,r,t):
        return 2.0*np.sum(self._Ak(self.eigenvalues,r,t))


class CylinderAnalytical2D:
    def __init__(self,BiotR,BiotZ):
        pass

class RectangularAnalytical1D:
    @staticmethod
    def transcendental(biot,x):
        return x*np.tan(x)-biot
    
    def __init__(self,BiotR,n_roots=100):
        self.biot = BiotR
        a,b = RootFinder.BolzandoRootCounterIntervals(self.f,1e-8,5e-2,n_roots)
        self.eigenvalues = np.array([opt.bisect(self.f,a[i],b[i]) for i in range(len(a))])
    def f(self,x):
        return x*np.tan(x) - self.biot
    def _Ak(self,root,x,t):
        return np.cos(root*x)*np.sin(root)*np.exp(-root**2*t)/(root + np.sin(root)*np.cos(root))
    def T(self,x,t):
        return np.sum(self._Ak(self.eigenvalues,x,t))
    def t(self,x,T):
        func = lambda t: self.T(r,t) - T
        x0 = 0.0
        if(self.biot < 0.1):
            x0 = -np.log(T)/self.biot/2.0
        else:
            x0 = 1.0
        a,b = RootFinder.searchInterval(func,x0)
        xopt = opt.bisect(func,a,b)
        return opt.newton(func,xopt)

class RectangularAnalytical2D:
    def __init__(self,BiotX,BiotY):
        pass

class RectangularAnalytical3D:
    def __init__(self,BiotX,BiotY,BiotZ):
        pass

def plot_invBiot():
    log_invBiot = np.linspace(-2.0,2.0,num=100)
    Tdim = np.logspace(-3,-0.1,base=10,num=10)

    inv_Biot = 10**log_invBiot
    solutions = [CylinderAnalytical1D(10**(-i)) for i in log_invBiot]
    for T in Tdim:
        t = np.array([j.t(0.0,T) for j in solutions])

        plt.plot(inv_Biot,t,label='T*(r=0) = {0:.2e}'.format(T))
    plt.legend()
    plt.xlabel('1/Bi')
    plt.ylabel('t*')
    plt.show()
if __name__=='__main__':
    plot_invBiot()



    
    