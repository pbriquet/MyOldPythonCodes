import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
import os

class SolutionPlotter:
    """
    This class is a matplotlib wrapper for typical profile solution plot.
    """
    def __init__(self,solutionObj):
        self.solution = solutionObj

class DirichletRectangular:
    """ 
    This is a class for analytical solution of Diffusion in d dimensions with constant Dirichlet condition at boundaries (1,1,1), and zero flux at center (0,0,0) (symmetry condition).
    \nAt Init, the user can define the number of dimensions (dim=3), number of eigenvalues at each dimnesion (n_eigen=100), and scale of body (a,b,c) (scale=[1.0,1.0,1.0])
    \nBefore calculating the solution, the user must fix a position with prepare_solution(coordinate=[0.0,0.0,0.0])
    \n- solution(t):\t Calculate for each instant. The function doesn't accept "t" arrays yet.
    \n- average(t):\t Calculate the averate field for each instant.
    \n- solution_center(t):\t Calculate the field at center position for each instant.
    """
    def __init__(self,dim=3,n_eigen=100,scale=[1.0,1.0,1.0]):
        self.mode = 'NeumannCenter'
        self.n_eigen = n_eigen  # Number of eigenvalues for solution (equal for each dimension) n^dim = number of coefficients
        self.dim = dim          # 1D, 2D, 3D
        self.scale = scale      # a, b, c Rectangular Lengths
        self._calculate_coefficients()
    def _calculate_coefficients(self):
        if(self.mode == 'NeumannCenter'): 
            # With dT/dx = 0 at center, solution is given by cos(λ)=0
            self.lambs = np.array([(m+0.5)*np.pi for m in range(self.n_eigen)]) # λ', β', γ' arrays are the same
        else:
            self.lambs = np.array([m*np.pi for m in range(self.n_eigen)])
        arrays = [self.lambs for k in range(self.dim)] # Just putting in a list depending on dim.
        # λ = (a/a)λ', β = (a/b)β', γ = (a/c)γ' arrays
        scaled_lambs = [self.lambs*self.scale[0]/self.scale[k] for k in range(self.dim)] 
        # Meshgrid of λ'i, β'j, γ'k
        self.mesh_l = np.array(np.meshgrid(*arrays,indexing='ij'))
        # Meshgrid of λi, βj, γk
        self.mesh_scaled_l = np.array(np.meshgrid(*scaled_lambs,indexing='ij'))
        # Calculating A_ijk = <f,g_ijk>/<g_ijk,g_ijk> with f = 1 and g_ijk = cos(λ'i*x)*cos(β'j*y)*cos(γ'k*z)
        self.mesh_A = np.power(2,self.dim)*np.prod(np.sin(self.mesh_l)/self.mesh_l,axis=0)
        # α_ijk = λi^2 + βj^2 + γk^2
        self.alpha = np.sum(np.power(self.mesh_scaled_l,2),axis=0)
        # Calculating A^2_ijk. Useful for average field.
        self.mesh_A2 = np.power(self.mesh_A,2)
    def prepare_position(self,coordinate):
        self.X = np.prod(np.cos(np.array([self.mesh_l[i]*coordinate[i] for i in range(self.dim)])),axis=0)
        self.prepared_coordinate = True
    # Solution is given by: sum_i(sum_j(sum_k(A_ijk*cos(λ'i*x)*cos(β'j*y)*cos(γ'k*z)*exp(-α_ijk*t))))
    def solution(self,time):
        tmp = np.sum(self.mesh_A*self.X*np.exp(-self.alpha*time))
        return tmp
    # Average is given by: 1/2^d*sum_i(sum_j(sum_k(A^2_ijk*exp(-α_ijk*t))))
    def average(self,time):
        temp = np.sum(self.mesh_A2*np.exp(-self.alpha*time))/np.power(2,self.dim)
        return temp
    # Solution at center is given by: sum_i(sum_j(sum_k(A_ijk*exp(-α_ijk*t))))
    def solution_center(self,time):
        tmp = np.sum(self.mesh_A*np.exp(-self.alpha*time))
        return tmp
class AnalyticalDiffusion:
    types_of_coordinate = ['Rectangular','Cylindrical','Spherical']
    types_of_conditions = ['Dirichlet','Neumann','Newton']
    def __init__(self,dim=1,coordinates='Rectangular',boundaryCondition='Dirichlet',initialProfile='Constant'):
        self.dim = dim
        self.coordinates = coordinates
        self.boundaryCondition = boundaryCondition
        self.initialProfile = initialProfile

if __name__=='__main__':
    plotter = 'plot_contour'

    b0 = dict(a=1.0,b=1.0,c=1.0) 
    b1 = dict(a=360.0e-3,b=360.0e-3,c=710.0e-3)
    b2 = dict(a=355.0e-3,b=355.0e-3,c=500.0e-3)
    b3 = dict(a=300.0e-3,b=300.0e-3,c=600.0e-3)
    b4 = dict(a=255.0e-3,b=255.0e-3,c=700.0e-3)

    D = np.linspace(1e-8,5.2e-8,num=100)
    #D = 5.2e-8
    bodies = [b1,b2,b3,b4]
    body = b1

    test = DirichletRectangular(dim=3,n_eigen=20,scale=[body['a'],body['b'],body['c']])
    vector = np.array([0.0,0.0,0.0])
    #time = 96.0*3600.0
    time = 200.0*3600.0
    positions = np.array([0.0] + list(np.linspace(0.1,1.0,num=9)))
    tau = (body['a']**2)/4.0/D/3600.0
    ave = []; sol = []
    
    
    columns = ['t','Average'] + ['x = ' + '{:1.1f}'.format(x) for x in positions]

    df = pd.DataFrame(columns=columns)
    _x = D
    _y = positions
    color=cm.rainbow(np.linspace(1,0,len(positions)))

    dimensionless_time = True
    mesh = [[],[],[]]
    for x in positions:
        test.prepare_position([x,0.0,0.0])
        sol.append([])
        for d in D:
            tau = (body['a']**2)/4.0/d
            mesh[0].append(d)
            mesh[1].append(x)
            sol[-1].append(test.solution(time/tau))
            mesh[2].append(sol[-1][-1])
        sol[-1] = np.array(sol[-1])
        #df['x = ' + '{:1.1f}'.format(x)] = sol[-1]
    

    #__location__= os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    #filepath = os.path.join(__location__,'b4.xlsx')
    #df.to_excel(filepath)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111,projection='3d')
    X = np.array(mesh[0]).reshape(len(_y),len(_x))
    Y = np.array(mesh[1]).reshape(len(_y),len(_x))
    Z = np.array(mesh[2]).reshape(len(_y),len(_x))
    norm = mp.colors.Normalize(vmin = 0.0, vmax = 1.0, clip = False)
    n_levels = 21
    #cont = ax.contourf(X,Y,Z,cmap='jet',levels=list(np.linspace(0.0,1.0,num=n_levels)))
    cont = ax.plot_surface(X,Y,Z,cmap='jet',norm=norm)
    ax.set_ylabel(r'$x^* = (x/a)$' + '\t' + r'$(x^*,0,0)$')
    ax.set_xlabel(r'$D$' + ' ' + r'$(m^2/s)$')
    ax.set_zlim(0.0,1.0)
    
    #title_line = r'$2a=$' + '{:1.1f}'.format(body['a']*1e3) + ' mm\t' + r'$2b=$' + '{:1.2f}'.format(body['b']*1e3) + ' mm\t' + r'$2c=$' + '{:1.2f}'.format(body['c']*1e3) + ' mm'
    #title_line += '\n' + r'$D=$' + '{:1.2e}'.format(D) + ' ' + r'$m^2/s$'
    #title_line += '\t' + r'$(\frac{b}{a})=$' + str(body['b']/body['a']) + ' ' + r'$\frac{c}{a}=$' + '{:1.2f}'.format(body['c']/body['a'])
    #ax.set_title(title_line)
    cbar = plt.colorbar(cont)
    cbar.ax.set_title(r"$H^*=\frac{H}{H_0}$")
    title_line = 't = ' + '{:2.1f}'.format(time/3600.0) + 'h' + '\n'
    title_line += r'$2a=$' + '{:1.1f}'.format(body['a']*1e3) + ' mm\t' + r'$2b=$' + '{:1.2f}'.format(body['b']*1e3) + ' mm\t' + r'$2c=$' + '{:1.2f}'.format(body['c']*1e3) + ' mm'
    ax.set_title(title_line)
    ax.view_init(30, 45)
    plt.show()
    #plt.xscale('log')
    
