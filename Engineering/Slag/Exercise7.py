from slag import *
from sum_slag import *
import numpy as np
from scipy.optimize import minimize

A = Slag(('CaO',0.1),('SiO2',0.3),('Al2O3',0.6))
B = Slag(('CaO',0.3),('SiO2',0.3),('Al2O3',0.4))
C = Slag(('CaO',0.7),('SiO2',0.1),('Al2O3',0.2))
D = Slag(('CaO',0.05),('SiO2',0.15),('Al2O3',0.8))

materia_prima = [A,B,C,D]
preco = [0.0,150.0,210.0,400.0]
x0 = [0.25,0.25,0.25,0.25]

def objective(x):
    return (x[0]*preco[0]+x[1]*preco[1]+x[2]*preco[2]+x[3]*preco[3])

def constraint1(x):
    return 1.0 - (x[0] + x[1] + x[2] + x[3])

def constraint2(x):
    return -1.0 + SumSlag(materia_prima,x).calculate_slag().Blf_CaAl
def constraint3(x):
    return 4.0 - SumSlag(materia_prima,x).calculate_slag().Blf_CaAl
def constraint4(x):
    return 0.7 - SumSlag(materia_prima,x).calculate_slag().oxides_dict['Al2O3']
def constraint5(x):
    return -0.15 + SumSlag(materia_prima,x).calculate_slag().oxides_dict['SiO2']

b = (0.0,1.0)
b_A = (0.2,1.0)
bnds = (b_A,b,b,b)
con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
con5 = {'type': 'ineq', 'fun': constraint5}
cons = [con1,con2,con3,con4,con5]
sol = minimize(objective,x0,method='SLSQP', bounds=bnds,constraints=cons,options={'disp':True})

print(sol)

s = SumSlag(materia_prima,sol.x)
sl = s.calculate_slag()
print(sl.PrintBasicity())
