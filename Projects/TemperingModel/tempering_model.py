import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas as pd 

R = 8.31

def A(dTdt):
    return 3.62905117*np.log(dTdt) + 22.37800241

def B(dTdt):
    return 1.56791610e4*np.log( np.log(dTdt) + 4 + 5.42841947)

def C(dTdt):
    return -0.01375479*np.log(dTdt) + 0.21057478

def H0(dTdt):
    return 0.0

class Model:
    def __init__(self,dTdt):
        self.a = A(dTdt)
        self.b = B(dTdt)
        self.c = C(dTdt)
        self.H0 = H0(dTdt)
        self.Hfar = 245.0

    def calculate(self,df,col):
        self.H = [self.H0]
        self.ksi = [0.0]
        t = list(df['t'])
        T = list(df[col])
        for j,v in enumerate(t):
            if(j!=0):
                self.ksi.append(self.ksi[j-1] + self.k(T[j])*(t[j] - t[j-1]))
        print(self.ksi)
    def k(self,T):
        return self.a*np.exp(-self.b/R/(T + 273.15))
    def ksi_function(self,ksi):
        return 
    
if __name__ == "__main__":
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    df = pd.read_excel(os.path.join(__loc__,'tempering.xlsx'))
    #print(df.head())
    T1 = 20.0
    T2 = 0.15
    T3 = 0.035

    m1 = Model(T1)
    m2 = Model(T2)
    m3 = Model(T3)
    #print(m1.a, m1.b, m1.c)
    #print(m2.a, m2.b, m2.c)
    #print(m3.a, m3.b, m3.c)

    m3.calculate(df,'P3')