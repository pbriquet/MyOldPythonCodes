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

pol_order = range(1,21)

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

    return x,y,a_FN_coefficients,b_FN_coefficients
    

def plot_lines(a,b,a_FN_coefficients,b_FN_coefficients,**kwargs):
    
    a_order = [[],[]]
    b_order = [[],[]]

    for i in pol_order:
        a_order[0].append(i)
        a_order[1].append(np.polyfit(FN,a_FN_coefficients,i))
        b_order[0].append(i)
        b_order[1].append(np.polyfit(FN,b_FN_coefficients,i))

    x_a = np.linspace(min(FN),max(FN),num=50)
    x_b = np.linspace(min(FN),max(FN),num=50)
    y_a = []
    y_b = []
    fig = plt.figure()
    
    plt.xlabel('FN')
    plt.ylabel('a')
    order_view = [15,16,17,18,19,20]
    j=0
    for i in order_view:
        y_a.append([polynomial(a_order[1][i-1],k) for k in x_a])
        plt.plot(x_a,y_a[-1], label='MMQ: Polinomio de Ordem ' + str(i))
        j+=1
    plt.scatter(FN,a_FN,label='Dados Boehler')
    plt.legend()
    fig = plt.figure()
    
    plt.xlabel('FN')
    plt.ylabel('b')
    j=0
    for i in order_view:
        y_b.append([polynomial(b_order[1][i-1],k) for k in x_b])
        plt.plot(x_b,y_b[-1], label='MMQ: Polinomio de Ordem ' + str(i))
        j+=1
    plt.scatter(FN,b_FN,label='Dados Boehler')
    plt.legend()

def plot_surface(x,y,a_FN_coefficients,b_FN_coefficients):
    
    a_order = [[],[]]
    b_order = [[],[]]
    
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
            error[2].append(np.log(np.sqrt(error_sum)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.reshape(error[0], (len(pol_order), len(pol_order)))
    Y = np.reshape(error[1], (len(pol_order), len(pol_order)))
    Z = np.reshape(error[2], (len(pol_order), len(pol_order)))
    surf = ax.plot_surface(X,Y,Z,cmap=cm.hot)
    ax.set_xlabel('Ordem Polinomio de a(FN)', fontsize=8)
    ax.set_ylabel('Ordem Polinomio de b(FN)', fontsize=8)
    ax.set_zlabel('Log(Erro)', fontsize=8)
    fig.colorbar(surf,shrink=0.8)

def plot_error_line(x,y,a_FN_coefficients,b_FN_coefficients):
    a_order = [[],[]]
    b_order = [[],[]]
    
    for i in pol_order:
        a_order[0].append(i)
        a_order[1].append(np.polyfit(FN,a_FN_coefficients,i))
        b_order[0].append(i)
        b_order[1].append(np.polyfit(FN,b_FN_coefficients,i))
    
    fn_files = []
    for i in FN:
        fn_files.append(open(os.path.join(__location__,"FN " + str(i) + ".txt"),'r'))
    
    error = [[],[]]
    for i in range(len(pol_order)):
        error_sum = 0.0
        for k in range(len(fn_files)):
            a = polynomial(a_order[1][i],FN[k])
            b = polynomial(b_order[1][i],FN[k])
            for w in range(len(x[1][k])):
                error_sum += (y[1][k][w] - (a*x[1][k][w] + b))**2
        error[0].append(a_order[0][i])
        error[1].append(np.log(np.sqrt(error_sum)))
    plt.plot(error[0],error[1])
    plt.xlabel('Ordem do Polinomio')
    plt.ylabel('Ln(Erro)')
    plt.xticks(pol_order)
if __name__=='__main__':
    x,y,a_FN,b_FN = read_files()
    plot_lines(x,y,a_FN,b_FN)
    #plot_error_line(x,y,a_FN,b_FN)
    #plot_graph(a,b,a_FN,b_FN)
    plot_surface(x,y,a_FN,b_FN)
    plt.show()