from math import *
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping

class Network:
    def __init__(self,training_set):
        pass
    def calculate(self,x):
        return paraboloid(x)

class TrainingSet:
    def __init__(self,input_data,output_data):
        self.input_dim = len(input_data)
        self.output_dim = 1
        self.input_data = input_data
        self.output_data = output_data

class MyBounds(object):
    def __init__(self, xmax=[5.0,5.0,5.0,5.0,5.0,5.0],xmin=[0.0,0.0,0.0,0.0,0.0,0.0]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self,**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def paraboloid(x):
    return 3.0 + (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2)

network = Network(None)

def objective(x):
    return network.calculate(x)

def constraint1(x):
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 1.0

def constraint2(x):
    return 2.0 - (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])

x0 = [0.5,0.5,0.5,0.0,0.0,0.0]
b = (0.0,10.0)
bnds = (b,b,b,b,b,b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
cons = [con1,con2]
#sol = minimize(objective,x0,method="Nelder-Mead", bounds=bnds,constraints=cons,options={'disp':True})
minimizer_kwargs = {"method":"COBYLA","constraints":cons}
mybounds = MyBounds()
sol = basinhopping(objective,x0,minimizer_kwargs=minimizer_kwargs,accept_test=mybounds)
print(sol)