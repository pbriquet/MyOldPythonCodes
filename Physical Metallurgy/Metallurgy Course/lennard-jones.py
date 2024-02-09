import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def V(epsilon,sigma,r):
    return 4.0*epsilon*(np.power(sigma/r,12) - np.power(sigma/r,6))

if __name__=='__main__':
    x = np.linspace(0.1,0.3,num=50)
    y = np.linspace(0.1,0.3,num=50)
    xx, yy = np.meshgrid(x,y,indexing='ij')
    r = np.sqrt(np.power(xx,2) + np.power(yy,2))
    z = V(10.0,0.2,r)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx,yy,z,cmap=cm.inferno)
    plt.show()

