from enum import IntEnum
from Synapse import *
from Initializers import *

class ConnectionMode(IntEnum):
    Complete = 0    # All neurons of source layer are connected to all neurons of target layer.
    OneOne = 1      # A neuron of source layer is connected to only one neuron of target layer. Numbers of neurons of each layer must be equal.
class Connector:
    def __init__(self,sourceLayer,targetLayer,**kwargs):
        self.synapses = []

        self.connectionMode = kwargs.get('connectionMode',ConnectionMode.Complete)
        self.weightInitializer = kwargs.get('weightsInitializer',InitializationFunctions.ConstantFunction(1.0))
        self.momentum = kwargs.get('momentum',0.07) # Momentum is best explained with the ball example

        self.sourceLayer = sourceLayer
        self.targetLayer = targetLayer

        # Add this connector to each layer (source and target)

        sourceLayer.targetConnectors.append(self)
        targetLayer.sourceConnectors.append(self)

        self.ConstructSynapses()
    def Initialize(self):
        if(self.weightInitializer is not None):
            self.weightInitializer.Initialize(self)
    def ConstructSynapses(self):
        # Just implement connection mode generating synapses
        if(self.connectionMode == ConnectionMode.Complete):
            for target_neuron in self.targetLayer.neurons:
                for source_neuron in self.sourceLayer.neurons:
                    self.synapses.append(Synapse(source_neuron,target_neuron,self))
                    
        elif(self.connectionMode == ConnectionMode.OneOne):
            for k in xrange(len(self.sourceLayer.neurons)):
                self.synapses.append(Synapse(self.sourceLayer.neurons[k],self.targetLayer.neurons[k],self))
    def Jitter(self,jitterLimit):
        for synapse in self.synapses:
            synapse.Jitter(jitterLimit)

    def __str__(self):
        tmp = 'Number of Synapses: ' + str(len(self.synapses)) + ', '
        tmp += 'Source Neurons: ' + str(self.sourceLayer.neuronCount) + ', '
        tmp += 'Target Neurons: ' + str(self.targetLayer.neuronCount) + ', '
        tmp += '\n\t\tConnection Mode: ' + str(self.connectionMode.name) + ', '
        tmp += 'Weight Initializer: ' + str(self.weightInitializer.__class__.__name__) + ', '
        tmp += 'Momentum: ' + str(self.momentum)
        return tmp