import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def build_toy_data(xlim=[-0.5,1.5],ylim=[-1.5,1.5],noise=0.05,Nx=50,Ny=50):
     x = np.linspace(xlim[0],xlim[1],num=Nx,endpoint=True)
     y = np.linspace(ylim[0],ylim[1],num=Ny,endpoint=True)
     xv = []
     yv = []
     zv = []
     for i in range(len(x)):
         for j in range(len(y)):
            xv.append(x[i])
            yv.append(y[j])
            zv.append( (x[i]**2 + y[j]**2) + np.random.uniform(low=-1.0,high=1.0)*noise)
     
     return np.array(xv),np.array(yv),np.array(zv)


if __name__=='__main__':
    x,y,z = build_toy_data()
    xn = pd.DataFrame(x)
    yn = pd.DataFrame(y)
    zn = pd.DataFrame(z)
    
    xn_norm = (xn - xn.mean())/(xn.max() - xn.min())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.scatter(xn_norm,yn,zn)
    plt.show()
