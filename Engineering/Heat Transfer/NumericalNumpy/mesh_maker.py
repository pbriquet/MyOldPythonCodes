from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm

class CoordinateSystem:
    @staticmethod
    def Rectangular(x):
        if(len(x)==1):
            return np.array([x[0]])
        elif(len(x)==2):
            return np.array([x[0],x[1]])
        return np.array([x[0],x[1],x[2]])
    @staticmethod
    def Cylindrical(x):
        if(len(x)==1):
            return np.array([x[0]])
        elif(len(x)==2):
            return np.array([x[0]*np.cos(x[1]),x[0]*np.sin(x[1])])

        return np.array([x[0]*np.cos(x[1]),x[0]*np.sin(x[1]),x[2]])

    @staticmethod
    def Spherical(x):
        if(len(x)==1):
            return np.array([x[0]])
        elif(len(x)==2):
            return np.array([x[0]*np.cos(x[1]),x[0]*np.sin(x[1])])

        return np.array([x[0]*np.cos(x[1])*np.sin(x[2]),x[0]*np.sin(x[1])*np.sin(x[2]),x[0]*np.cos(x[2])])

class MeshMaker:
    def __init__(self,dim,L0,Lx,nx,coordinates=CoordinateSystem.Rectangular):
        self.dim = dim
        kronecker = np.zeros((dim,dim),int)
        for i in range(dim):
            kronecker[i,i] = 1
        self.L0 = np.array(L0)
        self.Lx = np.array(Lx)
        self.nx = np.array(nx)  # Mesh
        self.grid = np.indices(nx)   # Mesh Grid of Coordinates
        dx =  np.array([x/y for x, y in zip(map(float, (self.Lx-self.L0)), map(int, self.nx))])   # Pythonic way for dividing one list for another list
        deltas = np.array([self.grid[i]*dx[i] for i in range(dim)])
        self.p = np.array([self.L0[i] + 0.5*dx[i] + deltas[i] for i in range(dim)])
        self.x_plus = np.array([[[self.p[i] + (-1)**u*dx[j]/2.0*kronecker[i,j] for u in range(2)] for j in range(dim)] for i in range(dim) ])
        self.x_minus = np.array([[[self.p[i] - (-1)**u*dx[j]/2.0*kronecker[i,j] for u in range(2)] for j in range(dim)] for i in range(dim) ])

        self.p = coordinates(self.p)
        self.x_minus = coordinates(self.x_minus)
        self.x_plus = coordinates(self.x_plus)

    def makeMesh(self):
        fig = plt.figure()
        if(self.dim==3):
            ax = fig.gca(projection='3d')
            ax.scatter(self.p[0],self.p[1],self.p[2],marker='o',s=20)
            ax.scatter(self.x_plus[0],self.x_plus[1],self.x_plus[2],marker='X',s=20)
            ax.scatter(self.x_minus[0],self.x_minus[1],self.x_minus[2],marker='>',s=20)
            #ax.set_xlim(0.0,Lx[0])
            #ax.set_ylim(0.0,Lx[1])
            #ax.set_zlim(0.0,Lx[2])
            plt.show()
        elif(self.dim == 2):
            ax = fig.gca(projection='3d')
            ax.scatter(self.p[0],self.p[1],marker='o',s=20)
            ax.scatter(self.x_plus[0],self.x_plus[1],marker='X',s=20)
            ax.scatter(self.x_minus[0],self.x_minus[1],marker='>',s=20)
            #ax.set_xlim(0.0,Lx[0])
            #ax.set_ylim(0.0,Lx[1])
            #ax.set_zlim(0.0,Lx[2])
            plt.show()

if __name__=='__main__':
    m = MeshMaker(3,[0.0,0.0,0.0],[1.0,1.0,1.0],[4,5,4],coordinates=CoordinateSystem.Rectangular)
    m.makeMesh()