from math import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import re
from scipy.optimize import curve_fit
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match

FN = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
pol_order_a = 20
pol_order_b = 20

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
    x = [[],[]]
    y = [[],[]]
    for k in range(len(fn_files)):
        i = 0
        x[0].append(FN[k])
        y[0].append(FN[k])
        x[1].append([])
        y[1].append([])
        for line in fn_files[k]:
            if(i!=0):
                tmp = [float(u) for u in re.findall(numbers_match,line)]
                x[1][k].append(tmp[0])
                y[1][k].append(tmp[1])
            i+=1
        a,b = np.polyfit(x[1][k],y[1][k],1)
        a_FN_coefficients.append(a)
        b_FN_coefficients.append(b)
        fn_files[k].close()

    a_order = [[],[]]
    b_order = [[],[]]
    pol_order = range(1,21)
    for i in pol_order:
        a_order[0].append(i)
        a_order[1].append(np.polyfit(FN,a_FN_coefficients,i))
        b_order[0].append(i)
        b_order[1].append(np.polyfit(FN,b_FN_coefficients,i))

    error = [[],[],[]]

    fn_files = []
    for i in FN:
        fn_files.append(open(os.path.join(__location__,"FN " + str(i) + ".txt"),'r'))
    for i in range(len(pol_order)):
        for j in range(len(pol_order)):
            error_sum = 0.0
            for k in range(len(fn_files)):
                a = polynomial(a_order[1][i],FN[k])
                b = polynomial(b_order[1][j],FN[k])
                for w in range(len(x[1][k])):
                    error_sum += (y[1][k][w] - (a*x[1][k][w] + b))**2
            error[0].append(a_order[0][i])
            error[1].append(b_order[0][j])
            error[2].append(np.sqrt(error_sum))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.reshape(error[0], (len(pol_order), len(pol_order)))
    Y = np.reshape(error[1], (len(pol_order), len(pol_order)))
    Z = np.reshape(error[2], (len(pol_order), len(pol_order)))
    ax.plot_surface(X,Y,Z)
    plt.show()
    

def plot_lines(a,b,a_FN,b_FN):
    x_a = np.linspace(min(FN),max(FN),num=50)
    y_a = [polynomial(a,k) for k in x_a]

    x_b = np.linspace(min(FN),max(FN),num=50)
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
     
if __name__=='__main__':
    read_files()
    #plot_lines(a,b,a_FN,b_FN)
    #plot_graph(a,b,a_FN,b_FN)
    #plot_surface(a,b,a_FN,b_FN)
    #plt.show()