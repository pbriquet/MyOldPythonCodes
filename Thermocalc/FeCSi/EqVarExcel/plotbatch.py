
from math import *
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys, os
import pandas as pd
import copy
import numpy as np
import random
import os, sys

if __name__=='__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    comps = pd.read_csv(os.path.join(__location__,'log.csv'),sep=';')
    print(comps.head())
    input()
    files = [f for f in os.listdir(__location__) if (os.path.join(__location__, f)).endswith('.xlsx')]
    for f in files:
        path = os.path.join(__location__,f)
        df = pd.read_excel(path)