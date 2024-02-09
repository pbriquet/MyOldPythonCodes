import marshal
import os
import dill as pickle
from LearningRateFunctions import LearningRateFunctions


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
path = os.path.join(__location__,'save.p')

class InitializationFunctionsTest:
    class ConstantFunction:
        def __init__(self,constant):
            self.constant = constant
        def Initialize(self,arg):
            if(arg.__class__.__name__ == 'Layer'):
                for neuron in arg.neurons:
                    neuron.bias = self.constant
            elif(arg.__class__.__name__ == 'Connector'):
                for synapse in arg.synapses:
                    synapse.weight = self.constant
    class ZeroFunction(ConstantFunction):
        def __init__(self):
            self.constant = 0.0
class LearningRateFunctionsTest:
    class LinearFunction:
        def __init__(self,initial_learning_rate,final_learning_rate):
            self.initialLearningRate = initial_learning_rate
            self.finalLearningRate = final_learning_rate
        def GetLearningRate(self,currentIteration,trainingEpochs):
            return self.initialLearningRate + (self.finalLearningRate - self.initialLearningRate) * currentIteration / trainingEpochs
    class HyperbolicFunction:
        def __init__(self,initial_learning_rate,final_learning_rate):
            self.initialLearningRate = initial_learning_rate
            self.finalLearningRate = final_learning_rate
        def GetLearningRate(self,currentIteration,trainingEpochs):
            return self.initialLearningRate + (self.finalLearningRate - self.initialLearningRate) * currentIteration / trainingEpochs
    class ExponentialFunction:
        def __init__(self):
            pass
class NetworkTest:
    def __init__(self,n_input,m_output):
        self.inputLayer = LayerTest(n_input)
        self.outputLayer = LayerTest(m_output)
        self.func = lambda x: x*x
    def __str__(self):
        tmp = 'Input: ' + str(self.inputLayer) + ' | Output: ' + str(self.outputLayer)
        return tmp
class LayerTest:
    def __init__(self,neuronCount,**kwargs):
        self.neurons = []
        self.neuronCount = neuronCount
        self.useFixedBiasValues = kwargs.get('useFixedBiasValues',False)
        #self.activationFunction = kwargs.get('activationFunction',ActivationFunctions.Identity)
        self.learningRateFunction = kwargs.get('learningRateFunction',LearningRateFunctions.LinearFunction(0.5,0.1))
        self.biasInitializer = kwargs.get('biasInitializer',InitializationFunctionsTest.ZeroFunction())
        self.targetConnectors = []
        self.sourceConnectors = []
    def __str__(self):
        return str(self.neuronCount)

n = NetworkTest(3,1)
pickle.dump( n, open( path, "wb" ) )
n_load = pickle.load( open( path, "rb" ) )

print n_load.inputLayer.learningRateFunction.GetLearningRate(1,100)