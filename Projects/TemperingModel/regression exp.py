from scipy import optimize
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd

R = 8.31
class Zhang:
    def functional(self,A,B,C):
        self.a = [A,B,C]
        return self
    def model(self,X):
        return Zhang.alfa(X[0],X[1],self.a)
    def model_ln(self,X):
        return Zhang.ln_ln_alfa(X[0],X[1],self.a)

    @staticmethod
    def alfa(t,T,a):
        return 1.0 - np.exp(-a[0]*np.exp(-a[1]/R/(T + 273.15))*np.power(t*3600.0,a[2]))
    @staticmethod
    def ln_ln_alfa(t,T,a):
        return np.log(a[0]) - a[1]/R/(T + 273.15) + a[2]*np.log(t*3600.0)
    def __init__(self,a):
        self.a = a

if __name__=='__main__':
    resfriamento = 'T3'

    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    df = pd.read_excel(os.path.join(__loc__,resfriamento + '.xlsx'))
    times = [2,4,8,16]
    Temps = [300,400,500,600]

    H0_T = {
        'T1':534.0,
        'T2':528.0,
        #'T2':534.0,
        'T3':497.0,
        'T4':415.0,
        'T4.1':395.0
    }

    H0 = H0_T[resfriamento]
    Hinf = 245.0

    #a0 = [5.0,5.0e3,0.5]
    df['alfa'] = (H0 - df['HRC'])/(H0 - Hinf)
    df['ln(1-alfa)'] = np.log(1.0 - df['alfa'])
    df['ln[-ln(1-alfa)]'] = np.log(-df['ln(1-alfa)'])
    
    T = np.linspace(300.0,600.0,num=50)
    t = np.linspace(2,16,num=50)
    a0 = [10.0,1e4,0.7]
    model = Zhang(a0)
    #f = lambda X,A,B,C: model.functional(A,B,C).model(X)
    #p, conv = curve_fit(f, (df['t'],df['T']), df['alfa'], a0, method='lm')
    f = lambda X,A,B,C: model.functional(A,B,C).model_ln(X)
    p, conv = curve_fit(f, (df['t'],df['T']), df['ln[-ln(1-alfa)]'], a0, method='lm')
    
    print(p)
    a = [np.power(p[0],1.0/p[2]),p[1]/p[2],p[2]]
    print(a)
    #print(np.power(p[0],1.0/p[2]),p[1]/p[2],p[2])
    #p = [3.0,5e4,0.5]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    tt,TT = np.meshgrid(t,T,indexing='ij')
    model = Zhang(p)
    
    ax.plot_surface(tt,TT,model.model([tt,TT]),cmap=cm.inferno)
    ax.scatter(df['t'],df['T'],df['alfa'])
    ax.set_xlabel('t (h)')
    ax.set_ylabel('T (°C)')
    ax.set_zlabel(r'$\alpha$' + ' (-)')
    ax.set_title(resfriamento)
    #ax.plot_surface(tt,TT,model.model_ln([tt,TT]),cmap=cm.inferno)
    #ax.scatter(df['t'],df['T'],df['ln[-ln(1-alfa)]'])

    plt.show()
