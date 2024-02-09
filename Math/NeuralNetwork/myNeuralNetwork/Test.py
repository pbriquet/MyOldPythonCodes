from Network import *
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
path = os.path.join(__location__,'save.txt')
savefile = open(path,'w')
# Initialization of a Network
n_input = 3 # R^n
m_output = 1    # R^m
neuralNetworkKwargs = {
    'trainingMethod':TrainingMethod.Supervised
    }
neuralNetwork = Network(n_input,m_output)   # R^n -> R^m
layers_data = []    # Layers data in dictionary

layers_data.append({    # 1st Layer
    'neurons':3,
    'activationFunction':ActivationFunctions.Logistic,
    'learningRateFunction':LearningRateFunctions.LinearFunction(0.5,0.25),
    'biasInitializer':InitializationFunctions.ZeroFunction()
    })
layers_data.append({    # 1st Layer
    'neurons':3,
    'activationFunction':ActivationFunctions.Logistic,
    'learningRateFunction':LearningRateFunctions.LinearFunction(0.5,0.25),
    'biasInitializer':InitializationFunctions.ZeroFunction()
    })


# All Connectors will be initialized by the same
connection_data = {
    'weightsInitializer':InitializationFunctions.RandomFunction(-1.0,1.0),
    'connectionMode':ConnectionMode.Complete
    }

neuralNetwork.BuildHiddenLayers(layersData=layers_data,connectionData=connection_data)
neuralNetwork.Initialize()

# Training Set
trainingSet = TrainingSet(n_input,m_output)
crossvalidationSet = TrainingSet(n_input,m_output)
x = np.array([[0.0,0.0,1.0],[1.0,-1.0,0.0],[1.0,0.0,1.0],[1.0,1.0,1.0]])
#x = np.array([[1]])
#x = np.array([[0],[1],[1],[0]])
y = np.array([[0.0],[1.0],[1.0],[0.0]])

samples = []
samples_unnormal = []
for i in xrange(y.shape[0]):
    samples.append(TrainingSample(x[i],y[i]))
    samples_unnormal.append(TrainingSample(x[i],y[i]))
    trainingSet.Add(samples[i])
    crossvalidationSet.Add(samples_unnormal[i])

training_args = {
    'trainingMethod':TrainingMethod.Supervised,
    'jitterNoiseLimit':1.0e-4,
    'jitterEpoch':0,
    'trainingEpochs':10000,
    'normalized':True}

neuralNetwork.setTrainingArgs(**training_args)

#print '\n' + str(neuralNetwork)
#neuralNetwork.printWeights()
neuralNetwork.Learn(trainingSet,initialize=True,verbose=True,verbose_epoch_interval=1000,print_error_interval=1000)
neuralNetwork.printWeights()
neuralNetwork.printBias()
neuralNetwork.Save(savefile)
savefile.close()
loadfile = open(path,'r')
loadneural = Network.Load(loadfile)
print loadneural
loadneural.printWeights()
loadneural.printBias()
i = 0
for j in crossvalidationSet.trainingSamples:
    print '\nInput is: ' + str(j.inputVector)
    print '\nNormalized Input is: ' + str(trainingSet.trainingSamples[i].inputVector)
    print 'Output is: ' + str(neuralNetwork.Run(j.inputVector,verbose=False))
    print 'Excepted Output is:' + str(j.outputVector)
    print 'Normalized Output is:' + str(trainingSet.trainingSamples[i].outputVector)
    i += 1
