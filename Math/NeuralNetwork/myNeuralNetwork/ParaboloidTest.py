from Network import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


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

# Initialization of a Network
n_input = 2 # R^n
m_output = 1    # R^m
neuralNetworkKwargs = {
    'trainingMethod':TrainingMethod.Supervised
    }
neuralNetwork = Network(n_input,m_output)   # R^n -> R^m
layers_data = []    # Layers data in dictionary

layers_data.append({    # 1st Layer
    'neurons':25,
    'activationFunction':ActivationFunctions.Logistic,
    'learningRateFunction':LearningRateFunctions.HyperbolicFunction(0.5,0.1),
    'biasInitializer':InitializationFunctions.RandomFunction(-1.0,1.0),
    'useFixedBias':False
    })

layers_data.append({    # 1st Layer
    'neurons':10,
    'activationFunction':ActivationFunctions.Logistic,
    'learningRateFunction':LearningRateFunctions.HyperbolicFunction(0.5,0.1),
    'biasInitializer':InitializationFunctions.RandomFunction(-1.0,1.0),
    'useFixedBias':False
    })





# All Connectors will be initialized by the same
connection_data = {
    'weightsInitializer':InitializationFunctions.RandomFunction(-1.0,1.0),
    'connectionMode':ConnectionMode.Complete,
    'momentum':0.07,
    }

neuralNetwork.BuildHiddenLayers(layersData=layers_data,connectionData=connection_data)
neuralNetwork.Initialize()

# Training Set
trainingSet = TrainingSet(n_input,m_output)

x, y = generate_points([-2.0,2.0],[-2.0,2.0],15,15)

samples = []
for i in xrange(len(y)):
    samples.append(TrainingSample(x[i],y[i]))
    trainingSet.Add(samples[i])

training_args = {
    'trainingMethod':TrainingMethod.Supervised,
    'jitterNoiseLimit':1.0e-4,
    'jitterEpoch':50,
    'trainingEpochs':1000,
    'normalized':True}


neuralNetwork.setTrainingArgs(**training_args)



#print '\n' + str(neuralNetwork)
#neuralNetwork.printWeights()
neuralNetwork.Learn(trainingSet,initialize=True,verbose=True,verbose_epoch_interval=10,print_error_interval=10)
#neuralNetwork.printWeights()
path = os.path.join(__location__,'SavedNetworks\\ParaboloidSave.txt')
savefile = open(path,'w')
neuralNetwork.Save(savefile)

points_x = []
points_y = []
points_z = []
points_z_true = []
for i in xrange(len(trainingSet.trainingSamples)):
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
