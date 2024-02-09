import numpy as np
import matplotlib.pyplot as plt

class RectangularDirichlet:
    def __init__(self,a,b,c,n_coefficients):
        self.a = a
        self.b = b
        self.c = c
        self.n_coeff = n_coefficients
        self.calculate_coefficients()
    def calculate_coefficients(self):
        self.count = np.array([m for m in range(1,self.n_coeff + 1)])
        alpha = (self.count + 0.5)*np.pi/self.a
        beta = (self.count + 0.5)*np.pi/self.b
        gamma = (self.count + 0.5)*np.pi/self.c

        self.m,self.n,self.p = np.meshgrid(self.count,self.count,self.count,indexing='ij')
        self.alpha,self.beta,self.gamma = np.meshgrid(alpha,beta,gamma,indexing='ij')
        self.C = 4.0*np.sin(self.alpha[:,:,:])*np.sin(self.beta[:,:,:])*np.sin(self.gamma[:,:,:])/(2.0*self.alpha[:,:,:] + np.sin(self.alpha[:,:,:]))/(2.0*self.beta[:,:,:] + np.sin(self.beta[:,:,:]))/(2.0*self.gamma[:,:,:] + np.sin(self.gamma[:,:,:]))
        
    def T(self,x,y,z,t):
        pass
    def Tc(self,t):
        x = self.a/2.0
        y = self.b/2.0
        z = self.c/2.0
        XYZ = np.sin(self.m[:,:,:]/2.0*np.pi)*np.sin(self.n[:,:,:]/2.0*np.pi)*np.sin(self.p[:,:,:]/2.0*np.pi)
        tau = np.exp(-(np.power(self.m[:,:,:]/self.a*np.pi,2)+np.power(self.m[:,:,:]/self.a*np.pi,2)+np.power(self.m[:,:,:]/self.a*np.pi,2))*t)
        
        calculation = np.sum(XYZ[:,:,:]*tau[:,:,:])

if __name__=='__main__':
    rec = RectangularDirichlet(1.0,1.0,1.0,200)
    print(np.sum(rec.C))

