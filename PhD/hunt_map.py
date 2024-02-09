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

class HuntMap:
    def __init__(self,A,m,dTn,nT):
        self.A = A
        self.m = m
        self.dTn = dTn
        self.nT = nT
    
    @staticmethod
    def R(A,m,dTn,nT,G,V):
        dTc = np.power(V/A,1.0/m)
        mask = (dTc > dTn)
        tmp = np.zeros(dTc.shape)
        tmp[mask] = A/G[mask]/V[mask]/(m + 1.0)*(np.power(dTc[mask],m+1) - np.power(dTn,m+1))
        return tmp
    @staticmethod
    def fe(A,m,dTn,nT,G,V):
        return nT*4.0*np.pi/3.0*np.power(HuntMap.R(A,m,dTn,nT,G,V),3)
    @staticmethod
    def f(A,m,dTn,nT,G,V):
        return 1.0 - np.exp(-HuntMap.fe(A,m,dTn,nT,G,V))

class DimensionlessHuntMap:
    def __init__(self,A,m,dTn,nT):
        self.A = A
        self.m = m
        self.dTn = dTn
        self.nT = nT
    
    @staticmethod
    def R(A,m,dTn,nT,G,V):
        dTc = np.power(V/A,1.0/m)
        mask = (dTc > dTn)
        tmp = np.zeros(dTc.shape)
        tmp[mask] = A/G[mask]/V[mask]/(m + 1.0)*(np.power(dTc[mask],m+1) - np.power(dTn,m+1))
        return tmp
    @staticmethod
    def fe(A,m,dTn,nT,G,V):
        return nT*4.0*np.pi/3.0*np.power(HuntMap.R(A,m,dTn,nT,G,V),3)
    @staticmethod
    def f(A,m,dTn,nT,G,V):
        return 1.0 - np.exp(-HuntMap.fe(A,m,dTn,nT,G,V))




if __name__=='__main__':
    
    A = 1e-4
    m = 2.0
    nT = 1e9
    fmin = 0.8
    fmax = 1.0
    
    log_G_min, log_G_max = [1.0,4.0]
    log_V_min, log_V_max = [-5.0,-3.0]
    n_G, n_V = [1000,1000]

    log_G = np.linspace(log_G_min,log_G_max,num=n_G)
    log_V = np.linspace(log_V_min,log_V_max,num=n_V)
    _dTn = np.linspace(0.0,15.0,num=20)
    _dTn = [0.0,0.5,0.75,1.25,2.25]
    _dTn.reverse()
    G,V = np.meshgrid(np.power(10.0,log_G),np.power(10.0,log_V))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    level_list = []
    fmt = {}
    lines = []
    for dTn in _dTn:
        fe = HuntMap.f(A,m,dTn,nT,G,V)
        
        idx = np.where(np.isnan(fe))
        fe[idx] = 0.0
        #fe = np.where((fe < fmax) & (fe > fmin),fe,0)
        cs = ax.contour(G,V,fe,levels=[0.49],colors='black')
        lines.append(cs.allsegs[0][0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    for k,v in enumerate(lines):
        ax.plot(v[:,0],v[:,1],color='black',label=r'$\Delta T_N = $' + str(_dTn[k]))
    labelLines(fig.gca().get_lines(),zorder=3.5)
    ax.set_xlim(10**log_G_min,10**log_G_max)
    ax.set_ylim(10**log_V_min,10**log_V_max)
       
    plt.show()