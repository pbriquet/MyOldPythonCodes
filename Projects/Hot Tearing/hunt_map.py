import numpy as np

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