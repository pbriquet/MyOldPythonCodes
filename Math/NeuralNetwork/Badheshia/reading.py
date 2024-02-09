import keras as K
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
Peet = 'PeetBadheshiaThermalConductivity.csv'
filepath = os.path.join(__location__,Peet)

peet_file = open(filepath,'r')
panda = pd.read_csv(peet_file,delimiter=';')

st = '0.15/0.35'


for i in panda['Si']:
    if(re.search("\.*/\.*",i)):
        x = i.split('/')
        print(x)
