from scipy.optimize import minimize

def objective(x):
    x1 = x[0]
    x2 = x[1]
    return -(x1 + x2)

def constraint1(x):
    x1 = x[0]
    x2 = x[1]
    return 20.0 - (2.0*x1 + 4.0*x2)

def constraint2(x):
    x1 = x[0]
    x2 = x[1]
    return 700.0 - (180.0*x1 + 100.0*x2)

def constraint3(x):
    return x[0]
def constraint(x):
    return x[0]

b = (0.0,400.0)
bnds = (b,b)
x0 = [1.0,2.0]
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
cons = [con1,con2]
sol = minimize(objective,x0,method='SLSQP', bounds=bnds,constraints=cons)

print sol
print constraint1(sol.x)
print constraint2(sol.x)
print constraint3(sol.x)