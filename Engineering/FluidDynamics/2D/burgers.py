import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import subprocess
import time

class Simulation:
    def __init__(self):
        self.nx, self.ny = [41,41]
        self.nt = 1200
        self.c = 1.0
        self.Lx, self.Ly = [2.0,2.0]

        self.dx, self.dy = [self.Lx/self.nx,self.Ly/self.ny]

        self.sigma = 0.0009
        self.nu = 2.0
        self.dt = self.sigma*self.dx*self.dy/self.nu
        self.initalize()

    def initalize(self):
        self.x = np.linspace(0,self.Lx,self.nx)
        self.y = np.linspace(0,self.Ly,self.ny)

        self.u = np.zeros((self.ny,self.nx))
        self.v = np.zeros((self.ny,self.nx))
        self.un = self.u.copy()
        self.vn = self.v.copy()

        self.comb = np.ones((self.ny,self.nx))

        self.Lxi, self.Lxf = [0.5,1.0]
        self.Lyi, self.Lyf = [0.5,1.0]

        u0, v0 = [3.0,3.0]
        j_min, j_max = [int(self.Lyi/self.dy),int(self.Lyf/self.dy + 1)]
        i_min, i_max = [int(self.Lxi/self.dx),int(self.Lxf/self.dx + 1)]
        self.u[j_min:j_max, i_min:i_max] = u0
        self.v[j_min:j_max, i_min:i_max] = v0
        self.steps = 0
    def boundary_conditions(self):
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] =0
        self.v[:, -1] = 0
    def RunSteps(self,nsteps=100):
        for n in range(nsteps):
            self.un = self.u.copy()
            self.vn = self.v.copy()
            self.u[1:-1, 1:-1] = (self.un[1:-1, 1:-1] -
                            self.dt / self.dx * self.un[1:-1, 1:-1] * 
                            (self.un[1:-1, 1:-1] - self.un[1:-1, 0:-2]) - 
                            self.dt / self.dy * self.vn[1:-1, 1:-1] * 
                            (self.un[1:-1, 1:-1] - self.un[0:-2, 1:-1]) + 
                            self.nu * self.dt / self.dx**2 * 
                            (self.un[1:-1,2:] - 2 * self.un[1:-1, 1:-1] + self.un[1:-1, 0:-2]) + 
                            self.nu * self.dt / self.dy**2 * 
                            (self.un[2:, 1:-1] - 2 * self.un[1:-1, 1:-1] + self.un[0:-2, 1:-1]))
            
            self.v[1:-1, 1:-1] = (self.vn[1:-1, 1:-1] - 
                            self.dt / self.dx * self.un[1:-1, 1:-1] *
                            (self.vn[1:-1, 1:-1] - self.vn[1:-1, 0:-2]) -
                            self.dt / self.dy * self.vn[1:-1, 1:-1] * 
                            (self.vn[1:-1, 1:-1] - self.vn[0:-2, 1:-1]) + 
                            self.nu * self.dt / self.dx**2 * 
                            (self.vn[1:-1, 2:] - 2 * self.vn[1:-1, 1:-1] + self.vn[1:-1, 0:-2]) +
                            self.nu * self.dt / self.dy**2 *
                            (self.vn[2:, 1:-1] - 2 * self.vn[1:-1, 1:-1] + self.vn[0:-2, 1:-1]))
            
            self.boundary_conditions()
            self.steps += 1 
        return self.u,self.v

if __name__=='__main__':
    s = Simulation()
    animation_step = 10
    frames = 10000
    X,Y = np.meshgrid(s.x,s.y)
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111)
    fig.set_label('u(x,y)')
    #ax.set_zlim(0.0,3.0)
    frame = None
    tstart = time.time()

    for i in range(frames):
        
        if frame:
            ax.collections.remove(frame)
        u,v = s.RunSteps(nsteps=animation_step) 
        frame = ax.quiver(X,Y,u[:],v[:],cmap=cm.hot)
        #frame = ax.plot_surface(X,Y,v[:],cmap=cm.hot)
        
        plt.pause(.001)




