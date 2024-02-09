from Helper import *

class InitializationFunctions:
    class ConstantFunction:
        def __init__(self,*args):
            self.constant = args[0]
        def Initialize(self,arg):
            if(arg.__class__.__name__ == 'Layer'):
                for neuron in arg.neurons:
                    neuron.bias = self.constant
            elif(arg.__class__.__name__ == 'Connector'):
                for synapse in arg.synapses:
                    synapse.weight = self.constant
        def __str__(self):
            return 'ConstantFunction\t' + str(self.constant)
                
    class NguyenWidrowFunction:
        def __init__(self,*args):
            self.outputRange = args[0]
        def Initialize(self,arg):
            if(arg.__class__.__name__ == 'Layer'):
                hiddenNeuronCount = 0
                for targetConnector in arg.targetConnectors:
                    hiddenNeuronCount += targetConnector.targetLayer.neuronCount
                nGuyenWidrowFactor = self.NGuyenWidrowFactor(arg.neuronCount,hiddenNeuronCount)
                for neuron in arg.neurons:
                    neuron.bias = Helper.GetRandom(-nGuyenWidrowFactor,nGuyenWidrowFactor)
            elif(arg.__class__.__name__ == 'Connector'):
                nGuyenWidrowFactor = self.NGuyenWidrowFactor(arg.SourceLayer.NeuronCount, arg.TargetLayer.NeuronCount)
                synapsesPerNeuron = arg.SynapseCount / arg.targetLayer.neuronCount
                
                for neuron in arg.targetLayer.neurons:
                    i = 0
                    normalizedVector = Helper.GetRandomVector(synapsesPerNeuron, nGuyenWidrowFactor)
                    for synapse in arg.GetSourceSynapses(neuron):
                        synapse.weight = normalizedVector[i]
                        i+=1

        def NGuyenWidrowFactor(self,inputNeuronCount,hiddenNeuronCount):
            return 0.7 * (hiddenNeuronCount)**(1.0 / inputNeuronCount) / self.outputRange
        def __str__(self):
            return 'NguyenWidrowFunction\t' + str(self.outputRange)

    class NormalizedRandomFunction:
        def __init__(self,*args):
            pass
        def Initialize(self,arg):
            if(arg.__class__.__name__ == 'Layer'):
                normalized = Helper.GetRandomVector(arg.neuronCount,1.0)
                i = 0
                for neuron in arg.layers:
                    neuron.bias = normalized[i]
                    i+=1
            elif(arg.__class__.__name__ == 'Connector'):
                normalized = Helper.GetRandomVector(arg.synapsesCount,1.0)
                i = 0
                for synapse in arg.synapses:
                    synapse.weight = normalized[i]
                    i += 1
        def __str__(self):
            return 'NormalizedRandomFunction'
    class RandomFunction:
        def __init__(self,*args):
            self.min_Limit = args[0]
            self.max_Limit = args[1]
        def Initialize(self,arg):
            if(arg.__class__.__name__ == 'Layer'):
                for neuron in arg.neurons:
                    neuron.bias = Helper.GetRandom(self.min_Limit,self.max_Limit)
            elif(arg.__class__.__name__ == 'Connector'):
                for synapse in arg.synapses:
                    synapse.weight = Helper.GetRandom(self.min_Limit,self.max_Limit)
        def __str__(self):
            return 'RandomFunction\t' + str(self.min_Limit) + '\t' + str(self.max_Limit)
    class ZeroFunction(ConstantFunction):
        def __init__(self,*args):
            self.constant = 0.0
        def __str__(self):
            return 'ZeroFunction'