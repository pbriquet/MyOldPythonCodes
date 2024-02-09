import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
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

class NewtonRectangular:
    """ 
    This is a class for analytical solution of Diffusion in d dimensions with Newton condition at boundaries (1,1,1), and zero flux at center (0,0,0) (symmetry condition).
    \nAt Init, the user can define the number of dimensions (dim=3), number of eigenvalues at each dimnesion (n_eigen=100), and scale of body (a,b,c) (scale=[1.0,1.0,1.0])
    \nBefore calculating the solution, the user must fix a position with prepare_solution(coordinate=[0.0,0.0,0.0])
    \n- solution(t):\t Calculate for each instant. The function doesn't accept "t" arrays yet.
    \n- average(t):\t Calculate the averate field for each instant.
    \n- solution_center(t):\t Calculate the field at center position for each instant.
    """
    def __init__(self,dim=3,Bi=[1.0,1.0,1.0],n_eigen=100,scale=[1.0,1.0,1.0]):
        self.mode = 'NeumannCenter'
        self.n_eigen = n_eigen  # Number of eigenvalues for solution (equal for each dimension) n^dim = number of coefficients
        self.dim = dim          # 1D, 2D, 3D
        self.scale = scale      # a, b, c Rectangular Lengths
        self.Bi = Bi
        self.calculate_roots()
        self._calculate_coefficients()
    def _calculate_roots(self):
        pass
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

class DirichletCylindrical:
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
    
    
