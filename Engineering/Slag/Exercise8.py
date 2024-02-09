# (pag. 210) O gerente da area de planejamento

from slag import *
from sum_slag import *
import numpy as np
from scipy.optimize import minimize

dross_aluminio = Slag(('CaO',0.03),('SiO2',0.05),('Al2O3',0.35),('Al',0.57))
aluminio_po = Slag(('Al2O3',0.02),('Al',0.98))
po_calcinacao = Slag(('CaO',0.85),('SiO2',0.12),('Al2O3',0.03))
cal_virgem = Slag(('CaO',0.95),('SiO2',0.03),('Al2O3',0.02))

materia_prima = [dross_aluminio,aluminio_po,po_calcinacao,cal_virgem]
preco = [450.0,1600.0,30.0,200.0]
x0 = [0.25,0.25,0.25,0.25]

def objective(x):
    return (x[0]*preco[0]+x[1]*preco[1]+x[2]*preco[2]+x[3]*preco[3])

def constraint1(x):
    return 1.0 - (x[0] + x[1] + x[2] + x[3])

def constraint2(x):
    return -7.0 + SumSlag(materia_prima,x).calculate_slag().Blf_CaAl


b = (0.0,1.0)
b_po = (0.2,1.0)
b_al = (0.05,1.0)
bnds = (b,b_al,b_po,b)
con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
cons = [con1,con2]
sol = minimize(objective,x0,method='SLSQP', bounds=bnds,constraints=cons,options={'disp':True})

# methods
# SLSQP
# BFGS
# CG
print sol

s = SumSlag(materia_prima,sol.x)
sl = s.calculate_slag()
print sl.PrintBasicity()
