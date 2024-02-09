import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Particle:
    def __init__(self,**kwargs):
        self.dT = kwargs.get('dT',0.0)
        self.R = 1e-9
        self.R_ = 0.0
        self.nv = kwargs.get('nv',1e8)
    
    @property
    def S(self):
        return 4.0*np.pi*np.power(self.R,2)
    @property
    def V(self):
        return 4.0*np.pi*np.power(self.R,3)/3.0

    @property
    def dV(self):
        return 4.0*np.pi*np.power(self.R,3)/3.0 - 4.0*np.pi*np.power(self.R_,3)/3.0
    
    @property
    def dfraction_exp(self):
        return self.dV*self.nv

    @property
    def dfraction(self):
        return np.exp(-self.nv*(4.0*np.pi*np.power(self.R,3)/3.0 - 4.0*np.pi*np.power(self.R_,3)/3.0))

    @property
    def fraction_ext(self):
        return self.V*self.nv

    @property
    def fraction(self):
        return np.exp(-self.fraction_ext)


    def grow(self,w,dt):
        self.R_ = self.R
        self.R += w*dt
    @staticmethod
    def growth(D,delta,C,Cfar):
        return D/delta*(C - Cfar)
if __name__=='__main__':
    tmax = 1e3
    dt = 1e-3
    t = 0
    p = Particle()
    
    while(t <= tmax):
        w = Particle.Growth(1e-9,1e-4,6.0,5.0)
        p.grow(w,dt)
        t += dt

    