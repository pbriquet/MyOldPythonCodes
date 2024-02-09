import numpy as np 
import random
if __name__ == "__main__":
    Lx = (5e-3,5e-3)
    nx = (5,5)
    dx = [L/nx[k] for k,L in enumerate(Lx)]

    x = np.linspace(0.0,Lx[0],num=nx[0])
    y = np.linspace(0.0,Lx[1],num=nx[1])

    xc = np.linspace(dx[0]/2.0,Lx[0] - dx[0]/2.0, num=nx[0])
    yc = np.linspace(dx[1]/2.0,Lx[1] - dx[1]/2.0, num=nx[1])

    mesh = np.array(np.meshgrid(xc,yc,indexing='ij'))
    activated = np.zeros(shape=mesh[0].shape,dtype=np.bool)
    liquid = np.ones(shape=mesh[0].shape,dtype=np.bool)
    substract = np.zeros(shape=mesh[0].shape,dtype=np.bool)
    substract_dT = np.zeros(shape=mesh[0].shape,dtype=np.float)
    pos_x = random.randint(0,nx[0] - 1)
    pos_y = random.randint(0,nx[1] - 1)

    substract[pos_x,pos_y] = 1
    substract_dT[pos_x,pos_y] = 0.1

    pos_x = random.randint(0,nx[0] - 1)
    pos_y = random.randint(0,nx[1] - 1)

    substract[pos_x,pos_y] = 1
    substract_dT[pos_x,pos_y] = 0.1
    Gx = 1e2
    Gy = 2e2

    dT = lambda X: Gx*X[0] + Gy*X[1]

    mask = np.ma.masked_array(mesh,mask=substract)
    print(mask)
    exit()
    dT_mesh = dT(mask)
    print(dT_mesh)
    exit()
    activate_nuclei = dT_mesh[substract] > substract_dT[substract]
    mx = np.ma.masked_array(substract,mask=activate_nuclei)
    print(mx)

