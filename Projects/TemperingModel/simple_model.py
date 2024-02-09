import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

class Model:
    def __init__(self):
        self.Qd = 5e4
        self.m = 0.5
        self.D0 = 3.0
        self.H0 = 40.0
        self.Hinf = 20.0
        self.R = 8.31

    def phi(self,t,T):
        return np.exp(-self.D0*np.exp(-self.Qd/self.R/T)*np.power(t,self.m))
    
    def H(self,t,T):
        return self.phi(t,T)*(self.H0 - self.Hinf) + self.Hinf
if __name__ == "__main__":
    m = Model()
    T = np.linspace(400.0 + 273.15,700.0 + 273.15,num=50)
    t = np.linspace(0,32,num=50)*3600

    tt,TT = np.meshgrid(t,T,indexing='ij')
    p = m.H(tt,TT) #+ np.random.rand(*tt.shape)*2.0

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    ax.plot_surface(tt,TT,p,cmap=cm.inferno)
    #ax.set_xscale('log')
    plt.show()

