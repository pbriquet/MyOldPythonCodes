import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os
def integrate(f_,x_):
    tmp = 0.0
    for k in range(len(x_)-1):
        tmp += (f_[k] + f_[k+1])*(x_[k+1] - x_[k])/2.0
    return tmp

class HotTearingScheilData:
    def __init__(self,folderpath):
        self.folderpath = folderpath
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for f in onlyfiles:
            if(f.endswith('.exp')):
class HotTearing:
    # fs_curve = list of (T,fs) 
    def __init__(self,fs_curves,betas):
        self.fs_curves = fs_curves
        self.betas = betas


    def calculateHCS(self,G,V,mu,lambda2):
        pass

if __name__=='__main__':
    Ts = 1100.0
    Tl = 1500.0
    beta = 0.087
    R = -1.0
    mu = 1e-3   # Pa.s
    G = 1e2     
    Vt = 1e-3
    lambda2 = 100.0e-6
    n = 20

    log_G_min, log_G_max = [1.0,4.0]
    log_V_min, log_V_max = [-5.0,-3.0]
    n_G, n_V = [1000,1000]

    log_G = np.linspace(log_G_min,log_G_max,num=n_G)
    log_V = np.linspace(log_V_min,log_V_max,num=n_V)
    _dTn = np.linspace(0.0,15.0,num=20)
    _dTn = [0.0,0.5,0.75,1.25,2.25]
    _dTn.reverse()
    G,V = np.meshgrid(np.power(10.0,log_G),np.power(10.0,log_V))

    _G = np.logspace(1,4,num=100)
    _Vt = np.logspace(-3,-1,num=100)
    x = np.linspace(0.0,(Tl-Ts)/G,num=n)
    T = np.linspace(Ts,Tl,num=n)

    fs = np.power((Tl - T)/(Tl - Ts),2.0)
    Gfs = fs/G

    i = integrate(fs,x)
    j = integrate(Gfs,T)
    print(i)
    print(j)



    '''
    fl = 1.0 - fs
    flvl = np.zeros(fl.shape)
    vl = np.zeros(fl.shape)
    flvl[-1] = -Vt*beta
    vl[-1] = -Vt*beta
    for k in range(len(flvl)-2,-1,-1):
        flvl[k] = flvl[k+1] - Vt*beta*(fs[k] - fs[k+1])
        vl[k] = flvl[k]/(1.0 - fs[k])

    print(vl)

    #f = -flvl*180.0*mu*fs**2/lambda2**2/(1.0-fs)**3

    #i = integrate(f,x)
    #print(x[-1])
    #print(i)
    '''