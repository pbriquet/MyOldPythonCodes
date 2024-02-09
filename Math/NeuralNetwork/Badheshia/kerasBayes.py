from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import normalize

import edward as ed
import numpy as np
import os
import re

from edward.models import Normal

class Normalizer:
    def __init__(self,data):
        self.max = data.max()
        self.min = data.min()
        self.mean = data.mean()
    
    def normalize(self,x):
        return (x - self.mean)/(self.max - self.min)
    def unnormalize(self,xn):
        return xn*(self.max - self.min) + self.mean

# Paraboloid data set
def build_toy_dataset(xlim=[-0.5,1.5],ylim=[-1.5,1.5],noise=0.05,Nx=50,Ny=50):
     x = np.linspace(xlim[0],xlim[1],num=Nx,endpoint=True)
     y = np.linspace(ylim[0],ylim[1],num=Ny,endpoint=True)
     xv = []
     yv = []
     zv = []
     for i in range(len(x)):
         for j in range(len(y)):
            xv.append(x[i])
            yv.append(y[j])
            zv.append( (x[i]**2 + y[j]**2) + np.random.uniform(low=-1.0,high=1.0)*noise)
     return np.array(xv),np.array(yv),np.array(zv)

def main():
    print("Hello")

if __name__ == "__main__":
    X,Y,Z = build_toy_dataset(xlim=[-3.0,3.0],ylim=[-1.0,3.0],noise=0.5,Nx=20,Ny=20)
    DataX = pd.DataFrame(X)
    DataY = pd.DataFrame(Y)
    DataZ = pd.DataFrame(Z)
    normalizer_x = Normalizer(DataX)
    normalizer_y = Normalizer(DataY)
    normalizer_z = Normalizer(DataZ)
    Xn = np.array([normalizer_x.normalize(i) for i in X])
    Yn = np.array([normalizer_y.normalize(i) for i in Y])
    Zn = np.array([normalizer_z.normalize(i) for i in Z])

    
    x = tf.placeholder(tf.float32, [None,2])
    hidden = Dense(25,activation='sigmoid')(x)
    w = Normal(loc=tf.zeros(2), scale=tf.ones(2))
    b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
    y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(1))
    z = tf.placeholder(tf.float32, [None,1])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X,Y,Z)
    ax.scatter(Xn,Yn,Zn)
    plt.show()
