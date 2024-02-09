from math import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import re
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
numbers = "1.28E+03 -4.5e-05 0.001"
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match

if __name__=='__main__':
    aws_a_points = open(os.path.join(__location__,"AWS a_points.txt"),'r')
    aws_b_points = open(os.path.join(__location__,"AWS b_points.txt"),'r')
    boehler_a_points = open(os.path.join(__location__,"Boehler a_points.txt"),'r')
    boehler_b_points = open(os.path.join(__location__,"Boehler b_points.txt"),'r')
    aws_a = [[],[]]
    aws_b = [[],[]]
    boehler_a = [[],[]]
    boehler_b = [[],[]]
    k = 0
    for line in aws_a_points:
        data = [float(x) for x in re.findall(numbers_match,line)]
        aws_a[0].append(data[0])
        aws_a[1].append(data[1])

    for line in aws_b_points:
        data = [float(x) for x in re.findall(numbers_match,line)]
        aws_b[0].append(data[0])
        aws_b[1].append(data[1])
    
    for line in boehler_a_points:
        data = [float(x) for x in re.findall(numbers_match,line)]
        boehler_a[0].append(data[0])
        boehler_a[1].append(data[1])
    
    for line in boehler_b_points:
        data = [float(x) for x in re.findall(numbers_match,line)]
        boehler_b[0].append(data[0])
        boehler_b[1].append(data[1])

    fig = plt.figure()
    plt.scatter(aws_a[0],aws_a[1],label='AWS')
    plt.scatter(boehler_a[0],boehler_a[1],label='Boehler')
    plt.xlabel('FN')
    plt.ylabel('a(FN)')
    plt.legend()

    fig = plt.figure()
    plt.scatter(aws_b[0],aws_b[1],label='AWS')
    plt.scatter(boehler_b[0],boehler_b[1],label='Boehler')
    plt.xlabel('FN')
    plt.ylabel('b(FN)')
    plt.legend()
    plt.show()