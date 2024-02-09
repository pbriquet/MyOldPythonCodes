from Network import *

neuralNetwork = Network(3,1)   # R^n -> R^m
layers_data = []    # Layers data in dictionary
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
training_args = {
    'trainingMethod':TrainingMethod.Supervised,
    'jitterNoiseLimit':1.0e-4,
    'jitterEpoch':0,
    'trainingEpochs':1,
    'normalized':True}
trainingSet = TrainingSet(3,1)
crossvalidationSet = TrainingSet(3,1)
x = np.array([[0.0,0.0,1.0],[10.0,-1.0,0.0],[1.0,0.0,1.0],[1.0,1.0,1.0]])
y = np.array([[0.0],[1.0],[1.0],[3.0]])
samples = []
neuralNetwork.setTrainingArgs(**training_args)


for i in xrange(y.shape[0]):
    samples.append(TrainingSample(x[i],y[i]))
    trainingSet.Add(samples[i])
    crossvalidationSet.Add(TrainingSample(x[i],y[i]))


#print trainingSet
neuralNetwork.Learn(trainingSet,initialize=True,verbose=True,verbose_epoch_interval=1,print_error_interval=1)
neuralNetwork.Run(x[0],verbose=True)
print trainingSet