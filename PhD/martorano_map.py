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

def create_ranges(start, stop,N, endpoint=True):
    if endpoint==1:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stop - start)
    return steps[:]*np.arange(N) + start[:]

def create_histogram(r,population,classes=30,percent=False,percent_cut=1e-8):
    dr = (r.max() - r.min())/classes
    count = []
    x = np.linspace(r.min(),r.max(),num=classes)
    average_x = []
    total = sum(population)
    for i,v in enumerate(x[:-1]):
        counter = 0
        for j,R in enumerate(r):
            if(R >= x[i] and R < x[i+1]):
                counter+=population[j]
        if(counter/total > percent_cut):
            count.append(counter)
            average_x.append((x[i] + x[i+1])/2.0)
    if(percent):
        count = count/total
    return np.array(average_x), np.array(count)


class MartoranoMap:
    @staticmethod
    def NormalDistribution(dTn,mu,sigma):
        if(sigma==0.0):
            return np.where(np.abs(dTn-mu)<=5e-4,mu,0.0)
        else:
            return 1.0/np.sqrt(2.0*np.pi)/sigma*np.exp(-1.0/2.0*((dTn-mu)/sigma)**2)
    
    @staticmethod
    def LogNormalDistribution(phi,mu,sigma):
        if(phi==0.0):
            return 0.0
        elif(sigma==0.0):
            return mu
        else:
            return 1.0/phi/np.sqrt(2.0*np.pi)/sigma*np.exp(-1.0/2.0*((np.log(phi/mu)/sigma)**2))
    
    @staticmethod
    def get_families(distribution,dTn):
        tmp = []
        for j,dT in enumerate(dTn[:-1]):
            tmp.append((distribution(dTn[j]) + distribution(dTn[j+1]))*(dTn[j+1] - dTn[j])/2.0)
        tmp.append(distribution(dTn[-1])*(dTn[-1] - dTn[-2])/2.0)
        return np.array(tmp)
    def __init__(self,mu=0.5,sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.distribution = lambda dT: MartoranoMap.NormalDistribution(dT,self.mu,self.sigma)
        self.m = 2.7
        self.A = 3e-6
        self.nt = 5e6

        #self.Gdim = 1.0/(np.power(self.nt,1.0/3.0)*self.mu)
        #self.Vdim = 1.0/(self.A*np.power(self.mu,self.m))
        #self.dTdim = 1.0/self.mu

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

def plot_map_contour():
    #m = MartoranoMap(mu=3.0,sigma=0.1)
    m = MartoranoMap(mu=3.0,sigma=0.0)
    log_G_min, log_G_max = [1.0,4.0]
    log_V_min, log_V_max = [-5.0,-2.0]
    n_G, n_V = [50,50]

    log_G = np.linspace(log_G_min,log_G_max,num=n_G)
    log_V = np.linspace(log_V_min,log_V_max,num=n_V)
    G = np.power(10.0,log_G)
    V = np.power(10.0,log_V)
    mG,mV = np.meshgrid(G,V)
    m_logG, m_logV = np.meshgrid(log_G,log_V)
    
    fraction = []


    for i in V:
        for j in G:
            fraction.append(m.calculate_fraction(j,i))
        print('V = ' + str(i))
    fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    ax = fig.add_subplot(111)
    z = np.array(fraction).reshape((len(G),len(V)))
    levels = [0.1*k for k in range(11)]
    #cs = ax.plot_surface(m_logG,m_logV,z,cmap=cm.magma)
    cs = ax.contourf(m_logG,m_logV,z,cmap=cm.magma,levels=levels)
    plt.colorbar(cs)
    plt.show()

def test():
    G = 1e3
    V = 1e-4
    m = MartoranoMap(mu=3.0,sigma=0.1)
    m.calculate_fraction(G,V)
if __name__=='__main__':
    plot_map_contour()