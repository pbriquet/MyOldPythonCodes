from Network import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import os,sys

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "SavedNetworks/ParaboloidSave.txt"
abs_file_path = os.path.join(script_dir, rel_path)

loadfile = open(abs_file_path,'r')


neuralNetwork = Network.Load(loadfile)

def paraboloid(x):
    return 3.0 + (x[0]**2 + x[1]**2)

def generate_points(x_limits,y_limits,n_x,n_y):
    dx = (x_limits[1] - x_limits[0])/n_x
    dy = (y_limits[1] - y_limits[0])/n_y
    x = []
    z = []
    n = 0
    for i in xrange(n_x + 1):
        for j in xrange(n_y + 1):
            x.append([x_limits[0] + i*dx, y_limits[0] + j*dy])
            z.append([paraboloid(x[n])])
            n += 1
    return x,z

trainingSet = TrainingSet(2,1)

x, y = generate_points([-2.0,1.0],[-2.0,1.0],50,50)

samples = []
for i in xrange(len(y)):
    samples.append(TrainingSample(x[i],y[i]))
    trainingSet.Add(samples[i])

points_x = []
points_y = []
points_z = []
points_z_true = []
for i in xrange(len(y)):
    points_x.append(x[i][0])
    points_y.append(x[i][1])
    points_z.append(neuralNetwork.Run([points_x[i],points_y[i]]))
    points_z_true.append(y[i])
#i = 0
#for j in trainingSet.trainingSamples:
#    print '\nInput is: ' + str(j.inputVector)
#    print 'Output is: ' + str(neuralNetwork.Run(j.inputVector,verbose=False))
#    print 'Excepted Output is:' + str(j.outputVector)
#    i += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_x,points_y,points_z,c='b')
ax.scatter(points_x,points_y,points_z_true,c='r')
plt.show()
