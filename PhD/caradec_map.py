from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm
import re, os
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import Normalize
from labellines import labelLine, labelLines
from haiderlevenspiel import HaiderLevenspiel

class CaradecMap:
    @staticmethod
    def NormalDistribution(dTn,mu,sigma):
        return 1.0/np.sqrt(2.0*np.pi)/sigma*np.exp(-1.0/2.0*((dTn-mu)/sigma)**2)
    
    @staticmethod
    def LogNormalDistribution(phi,mu,sigma):
        if(phi==0.0):
            return 0.0
        else:
            return 1.0/phi/np.sqrt(2.0*np.pi)/sigma*np.exp(-1.0/2.0*((np.log(phi/mu)/sigma)**2))
    
    @staticmethod
    def get_families(distribution,dTn):
        tmp = []
        for j,dT in enumerate(dTn[:-1]):
            tmp.append((distribution(dTn[j]) + distribution(dTn[j+1]))*(dTn[j+1] - dTn[j])/2.0)
        tmp.append(distribution(dTn[-1])*(dTn[-1] - dTn[-2])/2.0)
        return np.array(tmp)
    def __init__(self,distribution='instant',mu=0.5,sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.distribution = lambda dT: MartoranoMap.NormalDistribution(dT,self.mu,self.sigma)
        self.m = 2.7
        self.A = 3e-6
        self.nt = 5e6

        self.Gdim = 1.0/(np.power(self.nt,1.0/3.0)*self.mu)
        self.Vdim = 1.0/(self.A*np.power(self.mu,self.m))
        self.dTdim = 1.0/self.mu

    def growing_test(self):
        G = 1e3
        V = 1e-5
        d0 = 1e-7

        hl_args = {'rho_s':2700.0,'rho_f':2600.0,'g':9.81,'mu':1e-3}
        

        dTc = np.power(V/self.A,1.0/self.m)
        dy = 1e-5
        n_y = int(dTc/G/dy)
        y = np.linspace(0.0,dTc/G,num=n_y)
        dT = y*G
        _dT_contour = lambda _t,_y: _y*G
        dt = 1e-1
        n_t = int((dTc/G)/V/dt)
        t = np.linspace(0.0,(dTc/G)/V,num=n_t)
        print(n_t)
        w = lambda _dT: self.A*np.power(_dT,self.m)
        h = HaiderLevenspiel(shape='sphere',**hl_args)
        dT_pos = lambda _y: _y*G
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        

        y_pos = np.linspace(dTc/G,0.0,num=50)
        cmap = plt.get_cmap('magma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(y_pos))]
        for n,yn in enumerate(y_pos):
            p = [yn]
            d = [d0]
            v = lambda _d: -V - h.calculate(_d)
            v_save = [0.0]
            for j,_t in enumerate(t[1:]):
                p.append(p[-1] + (t[j+1] - t[j])*v(d[-1]))
                v_save.append(v(d[-1]))
                d.append(d[-1] + (t[j+1] - t[j])*2.0*w(dT_pos(p[-1])))

            ax.plot(t,p,d,color=colors[n-1])
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_zlabel('d')
        #mt,my = np.meshgrid(t,y)
        #mdT = _dT_contour(mt,my)
        
        
        #ax.plot_surface(mt,my,mdT)
        
        plt.show()

    def calculate_fraction(self,G,V):
        
        dTc = np.power(V/self.A,1.0/self.m)
        n = int(dTc/5e-4)
        dTn = np.linspace(0.0,dTc,n)
        curve = self.distribution(dTn)
        
        R = self.A/(1.0 + self.m)/G/V*(np.power(dTc,self.m+1.0) - np.power(dTn,self.m+1.0))

        R[np.where(R < 0.0)] = 0.0

        vol = 4.0*np.pi/3.0*np.power(R,3.0)

        #x,y = create_histogram(vol,curve,classes=100)
        #index = np.arange(len(x))
        #labels = ['{:0.2e}'.format(i) for i in x]
        #plt.bar(index,y)
        #plt.xticks(index, labels, fontsize=5,rotation=30)
        #plt.show()
        fraction = vol*curve
        fraction_total = self.nt*np.trapz(fraction,dTn)

        avrami = 1.0 - np.exp(-fraction_total)
        return avrami
    def plot_map(self):
        pass
    def plot_dimensionless_map(self):
        pass

if __name__=='__main__':
    c = CaradecMap()
    c.growing_test()