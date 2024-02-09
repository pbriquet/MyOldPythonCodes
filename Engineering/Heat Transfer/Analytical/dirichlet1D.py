import numpy as np
import matplotlib.pyplot as plt

class Dirichlet1D:
    def __init__(self,n_roots=100):
        self.n_roots = n_roots
        self.generate_coefficients()

    def generate_coefficients(self):
        self.lambs = np.array([(m+0.5)*np.pi for m in range(self.n_roots)])
        self.A = 2.0*np.array([np.sin(l_m)/l_m for l_m in self.lambs])
    
    def T(self,x,t):
        tmp = np.array([self.A[m]*np.cos(l_m*x)*np.exp(-l_m*l_m*t) for m,l_m in enumerate(self.lambs)])
        print(tmp)
        return np.sum(tmp)

class Dirichlet2D:
    def __init__(self,n_roots=100):
        self.n_roots = n_roots
        self.generate_coefficients()

    def generate_coefficients(self):
        self.lambs = np.array([(m+0.5)*np.pi for m in range(self.n_roots)])
        self.l_m,self.beta_n = np.meshgrid(self.lambs,self.lambs,indexing='ij')

        self.A = 4.0*np.sin(self.l_m)/self.l_m*np.sin(self.beta_n)/self.beta_n
        self.alpha = np.power(self.l_m,2)*np.power(self.beta_n,2)
    def T(self,x,y,t):
        tmp = self.A*np.cos(self.l_m*x)*np.cos(self.beta_n*x)*np.exp(-self.alpha*t)
        print(tmp)
        return np.sum(tmp)

class Dirichlet3D:
    def __init__(self,n_roots=100,a=1.0,b=1.0,c=1.0):
        self.n_roots = n_roots
        self.a = a; self.b = b; self.c = c
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
    test = Dirichlet3D(n_roots=100)
    t = np.linspace(0.0,0.5,num=100)
    T = []
    for _t in t:
        T.append(test.T(0.0,0.0,0.0,_t))
    plt.plot(t,T)
    plt.show()
    exit()
    x = 0.2
    t = np.linspace(0.0,10.0,num=10)

    Tm = [Am(m)*np.cos(lamb(m)*x)*np.exp(-np.power(lamb(m),2)*t) for m in range(10)]
    print(np.sum(Tm))

    Tm = [Am(m)*np.cos(lamb(m)*x)*np.exp(-np.power(lamb(m),2)*t) for m in range(100)]
    print(np.sum(Tm))

    Tm = [Am(m)*np.cos(lamb(m)*x)*np.exp(-np.power(lamb(m),2)*t) for m in range(1000)]
    print(np.sum(Tm))

    Tm = [Am(m)*np.cos(lamb(m)*x)*np.exp(-np.power(lamb(m),2)*t) for m in range(10000)]
    print(np.sum(Tm))