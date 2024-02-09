import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
from matplotlib import cm
import re, os
import pandas as pd
from functools import reduce

class Cubic:
    def __init__(self):
        self.angles = [90.0,90.0,90.0]

class Hexagonal:
    def __init__(self):
        self.angles = [90.0,90.0,90.0]

class UnitCell:
    def __init__(self,pos,size,structure='fcc'):
        self.pos = pos

class CrystalStructure:
    def __init__(self,a,b,c,alfa,beta,gamma):
        self.a = [a,b,c]
        self.alfa = [alfa,beta,gamma]

class Network:
    def __init__(self):
        self.n_cells = [5,5,5]
        self.total_cells = reduce(lambda x,y: x*y,self.n_cells)
        print(self.total_cells)
        self.L0 = [0.0,0.0,0.0]
        self.Lf = [1.0,1.0,1.0]
        self.dx = [(self.Lf[i] - self.L0[i])/self.n_cells[i] for i in range(3)]
        
        self.x = [[],[],[]]
        for i in range(3):
            self.x[i] = np.linspace(self.L0[i],self.Lf[i],num=self.n_cells[i])
        mx,my,mz = np.meshgrid(self.x[0],self.x[1],self.x[2])
        
        self.pos = [(i*self.dx[0],j*self.dx[0],k*self.dx[0]) for i in range(self.n_cells[0]) for j in range(self.n_cells[1]) for k in range(self.n_cells[2])]
        print(self.pos)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(mx,my,mz,s=500)
        plt.show()
        
        '''
        for i in range(3):
            self.pos.append([j*self.dx[i] for j in range(self.n_cells[i] + 1)])
        self.units = []
        for i in range(self.total_cells):
            self.units.append(UnitCell(pos[0][i]))
        '''
if __name__ == '__main__':
    n = Network() 