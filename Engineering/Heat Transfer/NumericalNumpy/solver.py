from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm
from mesh_maker import MeshMaker

if __name__=='__main__':

    m = MeshMaker(3,[0.0,0.0,0.0],[3e-2,3e-2,3e-2],[20,20,20])
    m.makeMesh()