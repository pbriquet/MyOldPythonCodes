from Helper import *
import math

class Synapse:
    def __init__(self,source_neuron,target_neuron,parent_connector):
        self.parent = parent_connector
        self.sourceNeuron = source_neuron
        self.targetNeuron = target_neuron
        self.sourceNeuron.targetSynapses.append(self)
        self.targetNeuron.sourceSynapses.append(self)
        self.weight = 1.0
        self.delta = 0.0
    def Backpropagate(self):
        self.sourceNeuron.error += self.targetNeuron.error * self.weight
    def Propagate(self):
        self.targetNeuron.input += self.sourceNeuron.output * self.weight
    def OptimizeWeight(self,learningFactor):
        tmp = self.delta * self.parent.momentum + learningFactor*self.targetNeuron.error * self.sourceNeuron.output
        if(not math.isnan(tmp)):
            self.delta = tmp
        self.weight += self.delta
    def Jitter(self,jitterLimit):   # Creates noise in weight values
        tmp = self.weight + Helper.GetRandom(-jitterLimit,jitterLimit)
        if(not math.isnan(tmp)):
            self.weight = tmp
    
