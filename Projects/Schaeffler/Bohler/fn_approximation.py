from math import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import re
import pandas as pd
from scipy.optimize import curve_fit,bisect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from labellines import labelLine, labelLines

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match

FN = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
pol_order_a = 14
pol_order_b = 14

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

def z_implicit(a,b,x,y):
    epsilon = 0.0005
    error = 1.0
    z = z_holder
    ITMAX = 5000
    i = 0
    while(error > epsilon and i < ITMAX):
        z_n = z - (y - polynomial(a,z)*x - polynomial(b,z))/(y - polynomial_derivative(a,z)*x - polynomial_derivative(b,z))
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

    x = np.linspace(x_min,x_max,num=50)
    y = np.linspace(y_min,y_max,num=50)
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
        line, = ax.plot(xp,yp[-1],color=colors[k],label=str(i))
        k+=1
    
    plt.xlabel(r'$Cr_{eq} = Cr + Mo + 0.7Nb$')
    plt.ylabel(r'$Ni_{eq} = Ni + 35C + 20N + 0.25Cu$')
    plt.title('WRC-1992 Ferrite Number')
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    labelLines(plt.gca().get_lines(),zorder=2.5)
def find_interval(a,b,x,y):
    FN_min = 0.5
    FN_max = 100.0
    dFN = 1.0
    FN = np.arange(FN_min,FN_max,dFN)
    for i in range(len(FN) - 1):
        angular = polynomial(a,FN[i])
        linear = polynomial(b,FN[i])
        f_a = y - (polynomial(a,FN[i])*x + polynomial(b,FN[i]))
        f_b = y - (polynomial(a,FN[i+1])*x + polynomial(b,FN[i+1]))
        if(f_a*f_b <= 0.0):
            return FN[i],FN[i+1]
    return None



def get_FNpoints(a,b):
    x_min = 17.0
    x_max = 31.0
    y_min = 9.0
    y_max = 18.0
    n_x,n_y = [50,50]
    Cr_min = 17.0
    Cr_max = 24.0
    Ni_min = 9.0
    Ni_max = 18.0
    rule = lambda _x,_y: _y - Ni_min - (Ni_max - Ni_min)/(Cr_max - Cr_min)*(_x - Cr_min)
    x = np.linspace(x_min,x_max,num=n_x)
    y = np.linspace(y_min,y_max,num=n_y)
    FN0 = 2.0
    FN_min = 0.5
    FN_max = 100.0
    points = [[],[],[]]
    for i in x:
        for j in y:
            #if(rule(i,j) >= 0.0):
            Cr_eq, Ni_eq = [i,j]
            FN_interval = find_interval(a,b,Cr_eq,Ni_eq)
            if(FN_interval != None):
                f = lambda x: Ni_eq - (polynomial(a,x)*Cr_eq + polynomial(b,x))
                FN = bisect(f,FN_interval[0],FN_interval[1])
                points[0].append(Cr_eq)
                points[1].append(Ni_eq)
                points[2].append(FN)
    df = pd.DataFrame({'Creq':points[0],'Nieq':points[1],'FN':points[2]})
    df.to_csv(os.path.join(__location__,'curve_points.csv'),sep=';')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(points[0],points[1],points[2])
    xdata = np.stack((points[0],points[1]))
    ydata = points[2]
    popt, pcov = curve_fit(curve2D, xdata, ydata)
    z_fit = curve2D(xdata, *popt)
    Z_fit = z_fit.reshape(len(points[0]))
    #ax.scatter(points[0],points[1],Z_fit,color='red')
    plt.show()

def poly_fit_try(x,A,B,C,D,E,F,G,H,I,J):
    return G*np.power(x[0],3) + H*np.power(x[1],3) + I*np.power(x[0],2)*x[1] + J*np.power(x[1],2)*x[0] + A*np.power(x[0],2) + B*np.power(x[1],2) + C*x[0]*x[1] + D*np.power(x[0],1) + E*np.power(x[0],1) + F
def curve2D(x, a, b,c,d,f):
    return a*np.power(x[0] - 31.0,3) + b*np.power(x[1] - 19.0,3) + c*(x[0] - 31.0) + d*(x[1] - 19.0) + f

def curve2Dplus(x, a, b,c,d,f):
    return a*np.power(x[0],2) + b*np.power(x[1],2) + c*(x[0]) + d*(x[1]) + f


def read_FNpoints(a,b):
    df = pd.read_csv(os.path.join(__location__,'curve_points.csv'),sep=';')
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(df['Creq'],df['Nieq'],df['FN'])

    xdata = np.stack((df['Creq'],df['Nieq']))
    ydata = np.log(df['FN'])
    popt, pcov = curve_fit(poly_fit_try, xdata, ydata)
    z_fit = poly_fit_try(xdata, *popt)
    z_fit = np.exp(z_fit)
    Z_fit = z_fit.reshape(len(ydata))
    ax.scatter(df['Creq'],df['Nieq'],Z_fit,color='red')
    print(popt)
    plt.show()

def FN3_Function(Cr_eq,Ni_eq):
    a = [2.87496272e+00,3.93235194e+00,-4.73343320e+00,5.44874587e+04,-5.45192272e+04,2.22443040e+02,-3.96737814e-02,-1.11572855e-01,3.30906987e-02,6.59212876e-02]
    x = [Cr_eq,Ni_eq]
    tmp = a[6]*np.power(x[0],3) + a[7]*np.power(x[1],3) + a[8]*np.power(x[0],2)*x[1] + a[9]*np.power(x[1],2)*x[0] + a[0]*np.power(x[0],2) + a[1]*np.power(x[1],2) + a[2]*x[0]*x[1] + a[3]*np.power(x[0],1) + a[4]*np.power(x[0],1) + a[5] 
    return tmp

def FN4_Function(Cr_eq,Ni_eq):
    a = [-1.27285636e-01,-1.91104461e-01,1.55403890e-01,5.73269004e+04,-5.73241225e+04,-2.29132546e+01,2.33243019e-03,8.99145111e-04,-5.47962610e-03,5.79044727e-03]
    x = [Cr_eq,Ni_eq]
    tmp = a[6]*np.power(x[0],3) + a[7]*np.power(x[1],3) + a[8]*np.power(x[0],2)*x[1] + a[9]*np.power(x[1],2)*x[0] + a[0]*np.power(x[0],2) + a[1]*np.power(x[1],2) + a[2]*x[0]*x[1] + a[3]*np.power(x[0],1) + a[4]*np.power(x[0],1) + a[5] 
    return np.exp(tmp)

def FN_Function(Cr_eq,Ni_eq):
    a = [ 4.72970336e-05,-1.22545113e-04,8.73897041e-01,-7.41594142e-01,6.85691024e+00]
    return np.exp(a[0]*np.power(Cr_eq - 31.0,3) + a[1]*np.power(Ni_eq - 19.0,3) + a[2]*(Cr_eq - 31.0) + a[3]*(Ni_eq - 19.0) + a[4])

def test_function():
    a = [2.05407541e-05,-5.32206387e-05,3.79528694e-01,-3.22070245e-01,2.97791849e+00]
    b = [-0.00380547,0.00399773,0.30391657,-0.22813185,-1.44609896]
    func = lambda _x: a[0]*np.power(_x[0] - 31.0,3) + a[1]*np.power(x[1] - 19.0,3) + a[2]*(_x[0] - 31.0) + a[3]*(_x[1] - 19.0) + a[4]
    func2 = lambda _x: b[0]*np.power(_x[0],2) + b[1]*np.power(x[1],2) + b[2]*(_x[0]) + b[3]*(_x[1]) + b[4]
    func3 = lambda _x: b[0]*np.power(_x[0],2) + b[1]*np.power(x[1],2) + b[2]*(_x[0]) + b[3]*(_x[1]) + b[4]
    x_min = 17.0
    x_max = 31.0
    y_min = 9.0
    y_max = 18.0
    n_x,n_y = [50,50]
    Cr_min = 17.0
    Cr_max = 24.0
    Ni_min = 9.0
    Ni_max = 18.0
    #rule = lambda _x,_y: _y - Ni_min - (Ni_max - Ni_min)/(Cr_max - Cr_min)*(_x - Cr_min)
    rule1 = lambda _x,_y: _y - 9.0 - (18.0 - 9.0)/(24.0 - 17.0)*(_x - 17.0)
    rule2 = lambda _x,_y: _y - 9.0 - (18.0 - 9.0)/(31.0 - 20.0)*(_x - 20.0)
    x = np.linspace(x_min,x_max,num=n_x)
    y = np.linspace(y_min,y_max,num=n_y)
    points = [[],[],[]]
    for i in x:
        for j in y:
            if(rule1(i,j) >= 0.0):
                Cr_eq, Ni_eq = [i,j]
                points[0].append(Cr_eq)
                points[1].append(Ni_eq)
                points[2].append(np.power(10.0,func([i,j])))
                '''
            elif(rule1(i,j) < 0.0 and rule2(i,j) >= 0.0):
                
                Cr_eq, Ni_eq = [i,j]
                points[0].append(Cr_eq)
                points[1].append(Ni_eq)
                points[2].append(np.power(10.0,func2([i,j])))
                '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(points[0],points[1],points[2])
    plt.show()

def plot_lines_3d(a,b):
    FN_numbers = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    FN_numbers = np.linspace(1.0,100.0,num=300)
    Creq_min, Creq_max = [17.0,31.0]
    Nieq_min, Nieq_max = [9.0,18.0]
    n_mesh = 200
    Cr_mesh = np.linspace(Creq_min,Creq_max,num=n_mesh)
    Ni_mesh = np.linspace(Nieq_min,Nieq_max,num=n_mesh)

    cmap = plt.get_cmap('magma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(FN_numbers))]
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for k,fn in enumerate(FN_numbers):
        angular = polynomial(a,fn)
        linear = polynomial(b,fn)
        x = Cr_mesh
        y = angular*Cr_mesh + linear
        mask = (y <= Nieq_max) & (y >= Nieq_min)
        n = len(y[mask])
        ax.plot(x[mask],y[mask],[fn]*n,color=colors[k])
        z_points = FN4_Function(x[mask],y[mask])
        error = abs((z_points - fn*n)/fn)
        
        #ax.scatter(x[mask],y[mask],z_points,c=error,cmap=cm.rainbow)
    ax.set_xlim(Creq_min,Creq_max)
    ax.set_ylim(Nieq_min,Nieq_max)
    plt.show()
if __name__=='__main__':
    a,b,a_FN,b_FN = read_files()
    #plot_lines(a,b,a_FN,b_FN)
    #get_FNpoints(a,b)
    #read_FNpoints(a,b)
    plot_graph(a,b,a_FN,b_FN)
    plt.show()
    #plot_surface(a,b,a_FN,b_FN)
    #plt.show()
    #test_function()
    #
    #print(FN4_Function(22.0,15.0))
    #plot_lines_3d(a,b)