from Synapse import *
from math import *
# A neuron has 'n' input synapses and 'm' output synapses
class Neuron:
    def __init__(self,parentLayer):
        self.input = 0.0
        self.output = 0.0
        self.error = 0.0
        self.bias = 0.0
        self.parentLayer = parentLayer
        self.sourceSynapses = []    # Each synapse if connect to a target synapse of the last neuron
        self.targetSynapses = []
    def EvaluateError(self):
        self.error = 0.0
        for synapse in self.targetSynapses:
            synapse.Backpropagate()
        self.error *= self.parentLayer.Derivative(self.input,self.output)
    def Run(self,verbose=False):
        self.value = 0.0
        tmp = ''
        for k in xrange(len(self.sourceSynapses)):
            self.sourceSynapses[k].Propagate()
            if(verbose):
                print '\t\tSynapse ' + str(k) + ': ' + str(self.sourceSynapses[k].sourceNeuron.output) + ' -> (w = ' + str(self.sourceSynapses[k].weight) + ') -> ' + str(self.sourceSynapses[k].weight*self.sourceSynapses[k].sourceNeuron.output)
        self.output = self.parentLayer.Activate(self.bias + self.input,self.output)
    def Output(self,_input):
        return self.output
    def Learn(self,learningRate):
        if(not self.parentLayer.useFixedBiasValues):
            self.bias += learningRate*self.error
        for synapse in self.sourceSynapses:
            synapse.OptimizeWeight(learningRate)