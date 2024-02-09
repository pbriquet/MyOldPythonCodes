import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm
import re, os
import pandas as pd

class Particle:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class Ensemble:
    def __init__(self,n):
        self.L0 = [0.0,0.0,0.0]
        self.Lf = [1.0,1.0,1.0]
        self.microcells = [100,100,100]
        self.macrocells = [2,2,2]
        self.dx = [(self.Lf[i] - self.L0[i])/self.microcells[i] for i in range(3)]
        self.dxM = [(self.Lf[i] - self.L0[i])/self.macrocells[i] for i in range(3)]
        pos = [[],[],[]]
        particles = []
        
        for j in range(int(n)):
            for i in range(3):
                pos[i].append(self.L0[i] + random.randint(0,self.microcells[i])*self.dx[i])
            particles.append(Particle(pos[0][-1],pos[1][-1],pos[2][-1]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(pos[0],pos[1],pos[2],alpha=0.2)
        ax.scatter(particles[0].x,particles[0].y,particles[0].z,c='red')
        plt.show()

if __name__=='__main__':
    e = Ensemble(1e4)