import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Model:
    def __init__(self):
        self.t = 0.0
    def runTime(self,tmax,dtmax,tmin=0.0,dtprint=10.0):
        self.dt = dtmax
        self.tmax = tmax
        self.tmin = tmin
        self.t = tmin
        self.dtprint = dtprint
        
        self.nx = [41,1,1]
        self.L = [2.0,1.0,1.0]
        self.dim = 1
        self.dx = [self.L[0]/self.nx[0],self.L[1]/self.nx[1],self.L[2]/self.nx[2]]

        self.u = np.ones(self.nx[0])
        self.u[int(.5 / self.dx[0]):int(1 / self.dx[0] + 1)] = 2 # Initial Conditions
        nu = 0.2
        un = np.ones(self.nx[0])
        while(self.t <= tmax):
            self.print()
            un = self.u.copy()

            for i in range(1,self.nx[0]-1):
                convective_term = un[i]*(un[i] - un[i-1])*self.dt/self.dx[0]
                diffusive_term = nu * self.dt / self.dx[0]**2 * (un[i+1] - 2.0 * un[i] + un[i-1])
                self.u[i] = un[i] - convective_term + diffusive_term
                        
            self.t+=self.dt
    def print(self):
        if((self.t - self.tmin)%self.dtprint < self.dt):
            plt.plot(np.linspace(0, 2, self.nx[0]), self.u,label=f't={self.t:.2f}')
        
        
if __name__=='__main__':
    m = Model()
    m.runTime(1.0,0.001,tmin=0.0,dtprint=0.1)
    plt.legend()
    plt.show()