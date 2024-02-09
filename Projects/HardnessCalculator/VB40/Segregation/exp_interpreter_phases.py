import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from Stencil import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import cm
import re

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
match = re.compile('[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?')

class exp_interpreter:
    def __init__(self,f):
        self.file = f
        self.compositions_T = dict()

    def read(self):
        phase = ' '
        reading_values = False
        clip = False
        log_file = open(os.path.join(__location__,'logfile.txt'),'w')
        for line in self.file:
            array = re.split('\W+',line)
            if(len(array)==3):
                    if(array[0]=='CLIP'):
                        if(array[1]=='ON'):
                            clip = True
                        elif(array[1]=='OFF'):
                            clip = False

            if(reading_values):
                if(len(array)>2):
                    if(array[1]=='PLOTTED'):
                        reading_values = False
                        clip = False
                    else:
                        array_number = [float(x) for x in re.findall(match, line)]
                        if(len(array_number)>0):
                            if(array[7] != 'MWA'):
                                self.compositions_T[phase].append((array_number[0],array_number[1]))

            if(not reading_values):
                if(len(array)>2):
                    if(array[1]=='PLOTTED'):
                        log_file.write("array = " + str(array) + "\n")
                        reading_values = True
                        phase = array[9].title() + "#" + array[10]
                        if(not self.compositions_T.has_key(phase)):
                            self.compositions_T.update({phase:[]})
                


#phases = ['cementite','ferrita','gamma','MC','liquid']
phases = ['phase_fraction']
files = []
phases_exp = dict()
for i in phases:
    f = open(os.path.join(__location__,i+".exp"),'r')
    files.append(f)
    e = exp_interpreter(f)
    e.read()
    phases_exp.update({i:e})

for k in phases_exp:
    for element in phases_exp[k].compositions_T:
        x_val = [x[0] for x in phases_exp[k].compositions_T[element]]
        y_val = [x[1] for x in phases_exp[k].compositions_T[element]]
        plt.plot(x_val,y_val,label=element)
plt.legend()
plt.show()