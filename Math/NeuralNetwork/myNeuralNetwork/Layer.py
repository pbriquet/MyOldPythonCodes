from Neuron import *
from ActivationFunctions import *
from LearningRateFunctions import *
from Initializers import *

class Layer:
    def __init__(self,neuronCount,**kwargs):
        self.neurons = []
        self.neuronCount = neuronCount
        self.useFixedBiasValues = kwargs.get('useFixedBiasValues',False)
        self.activationFunction = kwargs.get('activationFunction',ActivationFunctions.Identity)
        self.learningRateFunction = kwargs.get('learningRateFunction',LearningRateFunctions.LinearFunction(1.0,1.0))
        self.biasInitializer = kwargs.get('biasInitializer',InitializationFunctions.ZeroFunction())

        self.targetConnectors = []
        self.sourceConnectors = []

        for i in xrange(neuronCount):
            self.neurons.append(Neuron(self))
    def Initialize(self):
        if(self.biasInitializer is not None):
            self.biasInitializer.Initialize(self)
    def setInput(self,input_data):
        if(len(input_data) != len(self.neurons)):
            print "Error of size of input!"
            return
        for i in xrange(len(self.neurons)):
            self.neurons[i].input = input_data[i]
    def setOutput(self,output_data):
        if(len(output_data) != len(self.neurons)):
            print "Error of size of input!"
            return
        for i in xrange(len(self.neurons)):
            self.neurons[i].output = output_data[i]
    def getOutput(self):
        tmp = []
        for i in self.neurons:
            tmp.append(i.output)
        return tmp
    def setLearningRate(self,learningRate):
        self.learningRateFunction = LearningRateFunctions.LinearFunction(learningRate,learningRate)
    
    def setErrors(self,expectedOutput):
        if(len(expectedOutput) != len(self.neurons)):
            return "Error"
        meansquarederror = 0.0
        for i in xrange(len(self.neurons)):
            self.neurons[i].error = expectedOutput[i] - self.neurons[i].output
            meansquarederror += self.neurons[i].error**2
        return meansquarederror
    def EvaluateError(self):
        for neuron in self.neurons:
            neuron.EvaluateError()
    def Activate(self,_input,previousOutput):
        return self.activationFunction(_input,previousOutput)
    def Derivative(self,input_data,output_data):
        return self.activationFunction(input_data,output_data,deriv=True)
    def Learn(self,currentIteration,trainingEpochs):
        effectiveRate = self.learningRateFunction.GetLearningRate(currentIteration,trainingEpochs)
        for neuron in self.neurons:
            neuron.Learn(effectiveRate)
    def Run(self,verbose=False):
        tmp = ''
        for k in xrange(len(self.neurons)):
            if(verbose):
                print '\tNeuron ' + str(k) + ' starts:'
            self.neurons[k].Run(verbose=verbose)
            if(verbose):
                print '\tNeuron ' + str(k) + ': ' + str(self.neurons[k].input) + ' -> ' + self.activationFunction.__name__ + '  -> ' + str(self.neurons[k].output)
        if(verbose):
            print tmp
    def __str__(self):
        tmp = 'Number of Neurons: ' + str(self.neuronCount) + ', Activation Function: ' + self.activationFunction.__name__ + ', Learning Rate Function: ' + self.learningRateFunction.__class__.__name__ + ', Bias Initializer: ' + self.biasInitializer.__class__.__name__
        return tmp
        
