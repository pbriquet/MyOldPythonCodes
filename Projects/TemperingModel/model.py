import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

class Model:
    '''
    def __init__(self):
        self.Qd = 5e4
        self.m = 0.5
        self.D0 = 3.0
        self.H0 = 40.0
        self.Hinf = 20.0
        self.R = 8.31
    '''
    def __init__(self):
        self.Qd = 231e3
        self.m = 0.0518
        self.D0 = 2.7e8
        self.H0 = 776.0
        self.Hinf = 210.0
        self.R = 8.31
    '''
    def phi(self,t,T):
        return np.exp(-self.D0*np.exp(-self.Qd/self.R/(T+273.15))*np.power(3600*t,self.m))
    '''
    def phi(self,t,T):
        return np.exp(-np.power(self.D0*np.exp(-self.Qd/self.R/(T+273.15))*3600*t,self.m))
    def H(self,t,T):
        return self.phi(t,T)*(self.H0 - self.Hinf) + self.Hinf
if __name__ == "__main__":
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    
    data = {
        'T':[],
        't':[],
        'HRC':[]
    }
    model = Model()
    Temps = [300.0,400.0,500.0,600.0]
    times = [2,4,8,16]
    for T in Temps:
        for t in times:
            data['T'].append(T)
            data['t'].append(t)
            data['HRC'].append(model.H(t,T))
    df = pd.DataFrame(data=data,columns=data.keys())
    df.to_excel(os.path.join(__loc__,'data.xlsx'))

    print(df)