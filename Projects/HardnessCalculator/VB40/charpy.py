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
from HB_Data import *
from hardnessmodel import *
from scipy import optimize
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import cm
from hardnessmodel import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

files = []
files.append(open(os.path.join(__location__,"Impacto VB40.dat"),'r'))
files.append(open(os.path.join(__location__,"Impacto N6582.dat"),'r'))

for file in files:
    