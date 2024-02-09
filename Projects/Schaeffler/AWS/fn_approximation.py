from math import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import re
from scipy.optimize import curve_fit,bisect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match

FN = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
pol_order_a = 15
pol_order_b = 15

z_holder = 100.0


def polynomial(p,x):
    n = len(p)
    tmp = 0.0
    for i in range(n):
        tmp += p[i]*x**(n - i - 1)
    return tmp

def polynomial_derivative(p,x):
    n = len(p)
    tmp = 0.0
    for i in range(n):
        tmp += (n - i)*p[i]*x**(n - i - 2)
    return tmp

def read_files():
    fn_files = []
    for i in FN:
        fn_files.append(open(os.path.join(__location__,"FN " + str(i) + ".txt"),'r'))
    a_FN_coefficients = []
    b_FN_coefficients = []
    for k in range(len(fn_files)):
        i = 0
        x = []
        y = []
        for line in fn_files[k]:
            if(i!=0):
                tmp = [float(x) for x in re.findall(numbers_match,line)]
                x.append(tmp[0])
                y.append(tmp[1])
            i+=1
        a,b = np.polyfit(x,y,1)
        a_FN_coefficients.append(a)
        b_FN_coefficients.append(b)
    
    a = np.polyfit(FN,a_FN_coefficients,pol_order_a)
    b = np.polyfit(FN,b_FN_coefficients,pol_order_b)

    a_file = open(os.path.join(__location__,"a.txt"),'w')
    b_file = open(os.path.join(__location__,"b.txt"),'w')
    a_points_file = open(os.path.join(__location__,"a_points.txt"),'w')
    b_points_file = open(os.path.join(__location__,"b_points.txt"),'w')
    for i in range(len(FN)):
        a_points_file.write(str(FN[i]) + ";" + str(a_FN_coefficients[i]) + "\n")
        b_points_file.write(str(FN[i]) + ";" + str(b_FN_coefficients[i]) + "\n")
    k=0
    for i in a:
        a_file.write(str(k) + '\t' + str.format('{0:.15e}', i) + '\n')
        k+=1
    k=0
    for i in b:
        b_file.write(str(k) + '\t' + str.format('{0:.15e}', i) + '\n')
        k+=1
    return a,b,a_FN_coefficients,b_FN_coefficients

def plot_lines(a,b,a_FN,b_FN):
    x_a = np.linspace(min(FN),max(FN),num=100)
    y_a = [polynomial(a,k) for k in x_a]

    x_b = np.linspace(min(FN),max(FN),num=100)
    y_b = [polynomial(b,k) for k in x_b]

    fig = plt.figure()
    plt.plot(FN,a_FN,label='Dados DataThief')
    plt.plot(x_a,y_a, label='MMQ: Polinomio de Ordem ' + str(pol_order_a))
    plt.xlabel('FN')
    plt.ylabel('a')
    plt.title("Coeficiente Angular (a: f(x) = a*x + b)")
    plt.legend()

    fig = plt.figure()
    plt.plot(FN,b_FN,label='Dados DataThief')
    plt.plot(x_b,y_b,label='MMQ: Polinomio de Ordem ' + str(pol_order_b))
    plt.xlabel('FN')
    plt.ylabel('b')
    plt.title('Coeficiente Linear (b: f(x) = a*x + b)')
    plt.legend()

def z_implicit(a,b,x,y):
    epsilon = 0.0005
    error = 1.0
    z = z_holder
    ITMAX = 5000
    i = 0
    while(error > epsilon and i < ITMAX):
        z_n = z + (y - polynomial(a,z)*x - polynomial(b,z))/(polynomial_derivative(a,z)*x + polynomial_derivative(b,z))
        if(abs(z) > epsilon):
            error = abs((z_n - z)/z)
        else:
            error = abs(z_n-z)
        z = z_n
        i += 1
    if(i >= ITMAX):
        print("Exceeded")
    return z

def plot_surface(a,b,a_FN,b_FN):
    x_min = 21.0
    x_max = 26.0
    y_min = 11.0
    y_max = 14.0

    fig = plt.figure()

    x = np.linspace(x_min,x_max,num=20)
    y = np.linspace(y_min,y_max,num=20)
    X,Y = np.meshgrid(x,y)
    
    z = [[],[],[]]
    for i in x:
        for j in y:
            z[0].append(i)
            z[1].append(j)
            z[2].append(z_implicit(a,b,i,j))
            #z_holder = z[2][-1]
    #Z = z_implicit(a,b,X,Y)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z[0],z[1],z[2])
    #surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
    plt.show()
    



def plot_graph(a,b,a_FN,b_FN):
    x_min = 17.0
    x_max = 31.0
    y_min = 9.0
    y_max = 18.0
    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='--', linewidth=1)
    xp = np.linspace(x_min,x_max,num=50)
    yp = []
    cmap = plt.get_cmap('magma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(FN))]
    
    k = 0
    for i in FN:
        angular = polynomial(a,i)
        linear = polynomial(b,i)
        yp.append(angular*xp + linear)
        line, = ax.plot(xp,yp[-1],color=colors[k])
        k+=1
    
    plt.xlabel(r'$Cr_{eq}$')
    plt.ylabel(r'$Ni_{eq}$')
    plt.title('WRC Ferrite Number')
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))

def find_interval(a,b,x,y):
    FN_min = 0.5
    FN_max = 100.0
    dFN = 0.1
    FN = np.arange(FN_min,FN_max,dFN)
    for i in range(len(FN) - 1):
        angular = polynomial(a,FN[i])
        linear = polynomial(b,FN[i])
        f_a = y - (polynomial(a,FN[i])*x + polynomial(b,FN[i]))
        f_b = y - (polynomial(a,FN[i+1])*x + polynomial(b,FN[i+1]))
        if(f_a*f_b <= 0.0):
            return FN[i],FN[i+1]
    return None
def get_FNfunction(a,b):
    x_min = 17.0
    x_max = 31.0
    y_min = 9.0
    y_max = 18.0
    x = np.linspace(x_min,x_max,num=10)
    y = np.linspace(y_min,y_max,num=10)
    FN0 = 2.0
    FN_min = 0.5
    FN_max = 100.0
    for i in x:
        for j in y:
            Cr_eq, Ni_eq = [i,j]
            FN_interval = find_interval(a,b,Cr_eq,Ni_eq)
            if(FN_interval != None):
                f = lambda x: Ni_eq - (polynomial(a,x)*Cr_eq + polynomial(b,x))
                FN = bisect(f,FN_interval[0],FN_interval[1])
                print(Cr_eq, Ni_eq, FN)
if __name__=='__main__':
    a,b,a_FN,b_FN = read_files()
    #plot_lines(a,b,a_FN,b_FN)
    
    
    get_FNfunction(a,b)
    plot_graph(a,b,a_FN,b_FN)
    #plot_surface(a,b,a_FN,b_FN)
    plt.show()