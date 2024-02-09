from scipy.optimize import minimize

cost = [400.0,600.0,500.0]
x0 = [0.33,0.33,0.33]
def objective(x):
    return cost[0]*x[0]**2 + cost[1]*x[1] + cost[2]*x[2]

def constraint1(x):
    return x[0] - 3.0*x[1]

def constraint2(x):
    return 1.0 - (x[0] + x[1] + x[2])

b = (0.0,1.0)
bnds = (b,b,b)

con1 = {'type': 'eq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = [con1,con2]
sol = minimize(objective,x0,method='SLSQP', bounds=bnds,constraints=cons)

print sol

