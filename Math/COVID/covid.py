import numpy as np
import matplotlib.pyplot as plt 
from enum import IntEnum
import random

x_max = 1000
y_max = 1000

x_space = 10
y_space = 10 

u_ave = 2.0

class Disease:
    def __init__(self):
        self.probability_of_transmition = 0.9
        self.mortality = 0.06
        self.incubation_time = 2*7*24
        self.infection_rate = 2*7*24
    
    def Infect_Chance(self,infected,noninfected):



class Agent:
    states = ['Healthy','Infected','Imune','Dead']
    def __init__(self):
        self.state = 0
        self.pos = [random.randint(0,1000),random.randint(0,1000)]
    
    def interaction(self,other,disease):
            if(self.state == 0 and other.state == 0):
                pass 
            elif(self.state == 1 and other.state == 0):
                other.state = disease.infect_chance(self,other)
            elif(self.state == 0 and other.state == 1):
                other.state = disease.infect_chance(other,self)

        
    

class Arena:
    def __init__(self):
        self.x = np.linspace(0,x_max)
        self.y = np.linspace(0,y_max)
        self.xx, self.yy = np.meshgrid(x,y,indexing='ij')

if __name__=='__main__':
    pass 