import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def lambda_coeff(n,ratio_b_a=1.0):
    return np.array([(i + 0.5)*np.pi/ratio_b_a for i in range(1,n+1)])

class Dirichlet3D:
    def __init__(self,n_roots=100,body=b):
        self.n_roots = n_roots
        self.a = b['a']; self.b = b['b']; self.c = c['c']
        self.ratio_a = 1.0; self.ratio_b = self.b/self.a; self.ratio_c = self.c/self.a
        self.generate_coefficients()
        
    def generate_coefficients(self):
        self.lambs = np.array([(m+0.5)*np.pi for m in range(self.n_roots)])

        self.l_m,self.b_n,self.g_p = np.meshgrid(self.lambs/self.a,self.lambs/self.b,self.lambs/self.c,indexing='ij')
        
        self.A = 8.0*np.sin(self.l_m)/self.l_m*np.sin(self.b_n)/self.b_n*np.sin(self.g_p)/self.g_p
        self.alpha = np.power(self.l_m,2) + np.power(self.b_n,2) + np.power(self.g_p,2)
    def T(self,x,y,z,t):
        tmp = self.A*np.cos(self.l_m*x)*np.cos(self.b_n*y)*np.cos(self.g_p*z)*np.exp(-self.alpha*t)
        return np.sum(tmp)

if __name__=='__main__':

    T = 700.0 + 273.15
    b0 = dict(a=1.0,b=1.0,c=1.0) 
    b1 = dict(a=360.0,b=360.0,c=710.0)
    b2 = dict(a=355.0,b=355.0,c=500.0)
    b3 = dict(a=300.0,b=300.0,c=600.0)
    b4 = dict(a=255.0,b=255.0,c=700.0)
    bodies = [b1,b2,b3,b4]
    test = Dirichlet3D
    '''
    t = np.linspace(0.01,0.2,num=100)
    for k,body in enumerate(bodies):
        test = Dirichlet3D(n_roots=50,a=1.0,b=body['b']/body['a'],c=body['c']/body['a'])
    '''
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        T = []
        for _t in t:
            T.append(test.T(0.0,0.0,0.0,_t))
        ax.plot(t,T)
        ax.set_title(str(k))
        '''
    #plt.show()
    
    