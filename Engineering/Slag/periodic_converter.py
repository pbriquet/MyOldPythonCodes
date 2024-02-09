import sys, os  # Copy this heading to import from Libraries folders.
_libraries = ["Numerical","DesignPatterns"] # Here you can import Libraries from Libraries Folder. Just add to the list available folders on Libraries folder.
for _l in _libraries:
    s = sys.path[0].split("\\Projects\\",1)[0] + "\\Projects\\Libraries\\" + _l + "\\"
    sys.path.append(os.path.abspath(s))
# Error from this kind of import should be ignored


from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import ticker
from matplotlib import cm
import re


numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = re.compile(scientific_match + '|' + float_match)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

inputfile = (open(os.path.join(__location__,'periodic.txt'),'r'))
outputfile = (open(os.path.join(__location__,'periodic_python.txt'),'w'))
lines = []
for line in inputfile:
    s = re.sub('Elements.Add\(new Element\(','',line)
    l = re.findall(scientific_match + '|' + float_match + '|\w+',s)
    e = "self.elements['" + l[1] + "'] = Element('" + l[1] + "','" + l[2] + "'," + l[0] + "," + l[3] + ")\n"
    lines.append(e)
    outputfile.write(e)

inputfile.close()
outputfile.close()