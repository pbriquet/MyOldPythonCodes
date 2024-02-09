import numpy as np
from enum import IntEnum
from ActivationFunctions import *
from Neuron import *
from Layer import *
from Connection import *
from Helper import *
import re
from Initializers import *
import copy

'''
Teste
'''

class TrainingSample:
    ''' 
    A training sample is one point of data. 
    It is implemented as a class for checking.
    In future code, it will be just a wrapper for np.array
    '''
    def __init__(self,inputVector,outputVector):
        '''
        :inputVector: The input vector. Accepts a list, a number or a np.array
        '''
        Helper.ValidateNotNull(inputVector,'inputVector')
        Helper.ValidateNotNull(outputVector,'outputVector')
        self.inputVector = copy.copy(inputVector)
        self.outputVector = copy.copy(outputVector)
    def normalize_inputs(self,mu_inputs,sigma_inputs):
        '''
        This function receives a list of average (mu) values for the input array, and standard deviations (sigma)
        '''
        tmp = copy.copy(self.inputVector)
        for i in xrange(len(mu_inputs)):
            tmp[i] = (tmp[i] - mu_inputs[i])/sigma_inputs[i]
        return tmp
    def normalize_outputs(self,mu_outputs,sigma_outputs):
        '''
        This function receives a list of average (mu) values for the output array, and standard deviations (sigma)
        '''
        tmp = copy.copy(self.outputVector)
        for i in xrange(len(mu_outputs)):
            tmp[i] = (tmp[i] - mu_outputs[i])/sigma_outputs[i]
        return tmp

    def unnormalize_inputs(self,mu_inputs,sigma_inputs):
        '''
        This function receives a list of average (mu) values for the input array, and standard deviations (sigma)
        '''
        tmp = copy.copy(self.inputVector)
        for i in xrange(len(mu_inputs)):
            tmp[i] = (sigma_inputs[i]*tmp[i] + mu_inputs[i])
        return tmp
    def unnormalize_outputs(self,mu_outputs,sigma_outputs):
        '''
        This function receives a list of average (mu) values for the output array, and standard deviations (sigma)
        '''
        tmp = copy.copy(self.outputVector)
        for i in xrange(len(mu_outputs)):
            tmp[i] = (sigma_outputs[i]*tmp[i] + mu_outputs[i])
        return tmp

    def __str__(self):
        tmp = ''
        tmp += 'In: ' + str(self.inputVector) + ' Out: ' + str(self.outputVector)
        return tmp
class TrainingMethod(IntEnum):
    Supervised = 0
    Unsupervised = 1
class TrainingSet:
    def __init__(self,inputVectorLength,outputVectorLength):
        Helper.ValidatePositive(inputVectorLength,"inputVectorLength")
        Helper.ValidatePositive(outputVectorLength,"outputVectorLength")
        self.inputVectorLength = inputVectorLength
        self.outputVectorLength = outputVectorLength
        self.trainingSamples = []
        self.mu_inputs = [0.0]*inputVectorLength
        self.sigma_inputs = [1.0]*inputVectorLength
        self.mu_outputs = [0.0]*outputVectorLength
        self.sigma_outputs = [1.0]*outputVectorLength
    def Add(self,sample):
        self.trainingSamples.append(sample)
    @property
    def TrainingSampleCount(self):
        return len(self.trainingSamples)
    def NormalizeSamples(self,**kwargs):
        self.mu_inputs = []
        self.sigma_inputs = []
        if not bool(kwargs):
            for i in xrange(self.inputVectorLength):
                mu_tmp = 0.0
                for s in xrange(len(self.trainingSamples)):
                    mu_tmp += self.trainingSamples[s].inputVector[i]
                mu_tmp /= len(self.trainingSamples)
                sigma_tmp = 0.0
                for s in xrange(len(self.trainingSamples)):
                    sigma_tmp += (self.trainingSamples[s].inputVector[i] - mu_tmp)**2
                sigma_tmp /= len(self.trainingSamples)
                sigma_tmp = sqrt(sigma_tmp)
                self.mu_inputs.append(mu_tmp)
                self.sigma_inputs.append(sigma_tmp)
            for sample in self.trainingSamples:
                sample.inputVector = sample.normalize_inputs(self.mu_inputs,self.sigma_inputs)
            
            self.mu_outputs = []
            self.sigma_outputs = []
            for i in xrange(self.outputVectorLength):
                mu_tmp = 0.0
                for s in xrange(len(self.trainingSamples)):
                    mu_tmp += self.trainingSamples[s].outputVector[i]
                mu_tmp /= len(self.trainingSamples)
                sigma_tmp = 0.0
                for s in xrange(len(self.trainingSamples)):
                    sigma_tmp += (self.trainingSamples[s].outputVector[i] - mu_tmp)**2
                sigma_tmp /= len(self.trainingSamples)
                sigma_tmp = sqrt(sigma_tmp)
                self.mu_outputs.append(mu_tmp)
                self.sigma_outputs.append(sigma_tmp)
            for sample in self.trainingSamples:
                sample.outputVector = sample.normalize_outputs(self.mu_outputs,self.sigma_outputs)
        else:
            self.mu_inputs = kwargs['mu_inputs']
            self.sigma_inputs = kwargs['sigma_inputs']
            self.mu_outputs = kwargs['mu_outputs']
            self.sigma_outputs = kwargs['sigma_outputs']
            for sample in self.trainingSamples:
                sample.inputVector = sample.normalize_inputs(self.mu_inputs,self.sigma_inputs)
            for sample in self.trainingSamples:
                sample.outputVector = sample.normalize_outputs(self.mu_outputs,self.sigma_outputs)

    def __str__(self):
        tmp = 'Training Set:'
        k_samples = 0
        for k in self.trainingSamples:
            tmp += '\n\tSample ' + str(k_samples) + ': ' + str(k)
            k_samples += 1
        return tmp

    def __add__(self,other):
        tmp = TrainingSet(self.inputVectorLength,self.outputVectorLength)
        for sample in self.trainingSamples:
            tmp.Add(TrainingSample(copy.copy(sample.inputVector),copy.copy(sample.outputVector)))
        if(other is TrainingSet and (self.inputVectorLength == other.inputVectorLength and self.outputVectorLength == other.outputVectorLength)):
            for sample in other.trainingSamples:
                tmp.Add(TrainingSample(copy.copy(sample.inputVector),copy.copy(sample.outputVector)))
        return tmp

    

# Network: Defined by number of layers and number of neurons within
# Layer: Defined by a number of neurons and an activation function, with which each of its neurons will implement
# Connector: Defines how the a source layer is connected with target layer (complete connection or one by one)
# Connectors Generate the synapses between the sourceneurons and targetneurons
# Synapse: Provides the weight function between the source neuron output and the target neuron input. (w_j)

class Network:
    layersmodestart = 'Layer'
    connectionmodestart = 'Connection'
    weightmodestart = 'Weight'
    biasmodestart = 'Bias'
    # Function f : R^n -> R^m
    def __init__(self,n_input,m_output):
        self.n_input = n_input
        self.m_output = m_output
        self.inputLayer = Layer(n_input)
        self.outputLayer = Layer(m_output)

        self.layers = []    # List of Total Layers of the Network (includes input layer and output layer)
        self.hiddenlayers = []  # List of Hidden Layers. The Layers that define the operational of the Network.
        self.connectors = []    # Connectors are responsible for generating the synapses between neurons. Connectors are defined by a connection between two layers (source and target)
        self.TrainingMethod = TrainingMethod.Supervised

        self.jitterNoiseLimit = 1e-4 #  Maximum absolute limit to the random noise added during Jitter operation
        self.jitterEpoch = 0 # Epoch(interval) at which Jitter operation is performed. If this value is zero, not jitter is
        self.isStopping = False # This flag is set to true, whenever training needs to be stopped immmediately
    
        self.meanSquaredError = 0.0
        self.lastValidMeanSquaredError = 0.0
        self.meanSquaredErrorCV = 0.0

        self.mu_inputs = [0.0]*n_input
        self.sigma_inputs = [1.0]*n_input
        self.mu_outputs = [0.0]*m_output
        self.sigma_outputs = [1.0]*m_output

    def setTrainingArgs(self,**kwargs):
        self.TrainingMethod = kwargs.get('trainingMethod',TrainingMethod.Supervised)
        self.jitterNoiseLimit = kwargs.get('jitterNoiseLimit',1e-4)
        self.jitterEpoch = kwargs.get('jitterEpoch',0)
        self.trainingEpochs = kwargs.get('trainingEpochs',1000)
        self.normalized = kwargs.get('normalized',False)
        self.mu_inputs = kwargs.get('mu_inputs',[0.0]*self.n_input)
        self.sigma_inputs = kwargs.get('sigma_inputs',[1.0]*self.n_input)
        self.mu_outputs = kwargs.get('mu_outputs',[0.0]*self.m_output)
        self.sigma_outputs = kwargs.get('sigma_outputs',[1.0]*self.m_output)
    def BuildHiddenLayers(self,**kwargs):
        layers_data = kwargs.get('layersData',None)
        connection_data = kwargs.get('connectionData',None)
        if isinstance(layers_data,list):
            pass
        elif isinstance(layers_data,dict):
            layers_data = [kwargs.get('layersData',None)]
        else:
            layers_data = []

        numberOfLayers = len(layers_data)

        if isinstance(connection_data,list):
            pass
        elif isinstance(connection_data,dict):
            connection_data = [kwargs.get('connectionData',None)]
        else:
            connection_data = []
        
        numberOfConnections = len(connection_data)

        for i in xrange(numberOfLayers):
            self.layers.append(Layer(layers_data[i]['neurons'],**layers_data[i]))
            self.hiddenlayers.append(self.layers[i])
        self.layers.insert(0,self.inputLayer)
        self.layers.insert(len(self.layers),self.outputLayer)

        for i in xrange(len(self.layers) - 1):
            if(numberOfConnections == 0):
                self.connectors.append(Connector(self.layers[i],self.layers[i+1]))
            if(i < numberOfConnections):
                self.connectors.append(Connector(self.layers[i],self.layers[i+1],**connection_data[i]))
            else:
                self.connectors.append(Connector(self.layers[i],self.layers[i+1],**connection_data[numberOfConnections-1]))
            
    def Initialize(self):
        for layer in self.layers:
            layer.Initialize()
        for connector in self.connectors:
            connector.Initialize()

    def StopLearning(self):
        self.isStopping = True

    def _ResetValues(self):
        for k in self.layers:
            for j in k.neurons:
                j.input = 0.0
                j.output = 0.0
    # Main function of calculation of the network output.
    def Run(self,input_data,verbose=False):
        self._ResetValues()
        normalized_sample = TrainingSample(copy.copy(input_data),0.0)
        normalized_input = normalized_sample.normalize_inputs(self.mu_inputs,self.sigma_inputs)
        if(verbose):
            print "Input: " + str(input_data)
            print "Normalized Input: " + str(normalized_input)
            print "Mu inputs: " + str(self.mu_inputs) + " Sigma Inputs: " + str(self.sigma_inputs)
        self.inputLayer.setInput(normalized_input)
        for k in xrange(len(self.layers)):
            if(verbose):
                print 'Layer ' + str(k) + ':'
            self.layers[k].Run(verbose=verbose)
        normalized_sample.outputVector = copy.copy(self.outputLayer.getOutput())
        unnormal_output = normalized_sample.unnormalize_outputs(self.mu_outputs,self.sigma_outputs)
        return unnormal_output

    def __call__(self,arg): # Call method returning Run function.
        return self.Run(arg)

    def setLearningRate(self,learningRate):
        for k in self.hiddenlayers:
            k.setLearningRate(learningRate)

    # TrainingSet is the set of Training Samples. Training Epochs are the number of cycles of training the network. Initialize builds the initial parameters provided by an initialization function.
    def Learn(self,trainingSet,crossValidationSet=None,initialize=False,verbose=False,verbose_epoch_interval=100,print_error_interval=100):
        self.meanSquaredError = 0.0
        self.lastValidMeanSquaredError = 0.0
        self.isStopping = False
        self.trainingSet = trainingSet
        self.crossValidationSet = crossValidationSet
        if(self.normalized):
            all_samples = trainingSet + crossValidationSet
            all_samples.NormalizeSamples()
            self.mu_inputs = all_samples.mu_inputs
            self.sigma_inputs = all_samples.sigma_inputs
            self.mu_outputs = all_samples.mu_outputs
            self.sigma_outputs = all_samples.sigma_outputs
            trainingSet.NormalizeSamples(mu_inputs=self.mu_inputs,sigma_inputs=self.sigma_inputs,mu_outputs=self.mu_outputs,sigma_outputs=self.sigma_outputs)
            if(crossValidationSet != None):
                crossValidationSet.NormalizeSamples(mu_inputs=self.mu_inputs,sigma_inputs=self.sigma_inputs,mu_outputs=self.mu_outputs,sigma_outputs=self.sigma_outputs)
        if(initialize):
            self.Initialize()
        for currentIteration in xrange(self.trainingEpochs):
            if(verbose):
                if(currentIteration%verbose_epoch_interval == 0):
                    print 'Iteration = ' + str(currentIteration)
            randomOrder = Helper.GetRandomOrder(self.trainingSet.TrainingSampleCount)
            self.OnBeginEpoch()

            # Jitter!
            if(currentIteration > 0 and self.jitterEpoch > 0 and currentIteration % self.jitterEpoch == 0):
                if(verbose):
                    print "Jittering!"
                for connector in self.connectors:
                    connector.Jitter(self.jitterNoiseLimit)
            
            for k_sample in randomOrder:
                randomSample = self.trainingSet.trainingSamples[k_sample]

                self.OnBeginSample()
                self.LearnSample(randomSample,currentIteration,self.trainingEpochs)
                self.OnEndSample()

                if(self.isStopping):
                    self.isStopping = False
                    return

            self.OnEndEpoch()
            if(verbose):
                if(currentIteration%print_error_interval==0):
                    print 'Error: ' + str(self.meanSquaredError)
            if(self.isStopping):
                self.isStopping = False
                return

    def LearnSample(self,sample,currentIteration,trainingEpochs):
        self._ResetValues()
        self.inputLayer.setInput(sample.inputVector)
        for layer in self.layers:
            layer.Run()
        self.meanSquaredError += self.outputLayer.setErrors(sample.outputVector)

        for k in reversed(xrange(len(self.hiddenlayers))):
            self.hiddenlayers[k].EvaluateError()

        for layer in self.layers:
            layer.Learn(currentIteration,trainingEpochs)
    def getCrossValidationError(self):
        if(self.crossValidationSet != None):
            cvError = 0.0
            for sample in self.crossValidationSet.samples:
                self.inputLayer.setInput(sample.inputVector)
                for layer in self.layers:
                    layer.Run()
                cvError += self.outputLayer.setErrors(sample.outputVector)
            return cvError / self.crossValidationSet.TrainingSampleCount
        else:
            return 0.0
    # Events
    def OnBeginEpoch(self):
        self.lastValidMeanSquaredError = self.meanSquaredError
        self.meanSquaredError = 0.0
    def OnEndEpoch(self):
        self.meanSquaredError = self.meanSquaredError / self.trainingSet.TrainingSampleCount
        self.lastValidMeanSquaredError = self.meanSquaredError
    def OnBeginSample(self):
        pass
    def OnEndSample(self):
        pass
    def printBias(self):
        k_layer = 0
        for layer in self.layers:
            k_neuron = 0
            print '\nLayer ' + str(k_layer) + ':'
            for neuron in layer.neurons:
                print '\tNeuron ' + str(k_neuron) + ' Bias:' + str(neuron.bias) 
                k_neuron += 1
            k_layer += 1
    def printWeights(self):
        k_layer = 0
        for layer in self.layers:
            k_neuron = 0
            print '\nLayer ' + str(k_layer) + ':'
            for neuron in layer.neurons:
                print '\tNeuron ' + str(k_neuron) + ':'
                k_sourcesynapse = 0
                k_targetsynapse = 0
                #print '\tSources:'
                #for sourceSynapse in neuron.sourceSynapses:
                #    print '\t\tSynapse ' + str(k_sourcesynapse) + ': ' + str(sourceSynapse.weight) 
                #    k_sourcesynapse += 1
                print '\t\tTargets:'
                for targetSynapse in neuron.targetSynapses:
                    print '\t\tSynapse ' + str(k_targetsynapse) + ': ' + str(targetSynapse.weight) 
                    k_targetsynapse += 1
                k_neuron += 1
            k_layer += 1
    def __str__(self):
        tmp = '-- Network Configuration --\n'
        tmp += '\tTraining Method: ' + str(self.TrainingMethod.name) + '\n'
        tmp += '\tJist Limit Noise: ' + str(self.jitterNoiseLimit) + '\n'
        tmp += '\tJist Epoch: ' + str(self.jitterEpoch) + '\n'
        tmp += '\n-- Layers --\n'
        for k in xrange(len(self.layers)):
            if(k == 0):
                tmp += "Input "
            elif(k == len(self.layers) - 1):
                tmp += "Output "
            tmp += '\tLayer ' + str(k) + ': ' + str(self.layers[k]) + '\n'

        tmp += '\n-- Connectors --\n'
        for k in xrange(len(self.connectors)):
            tmp += '\tConnector ' + str(k) + ': ' +str(self.connectors[k]) + '\n'
        return tmp

    @staticmethod
    def Load(txt_file):
        neuraldatamode = True
        layerdatamode = False
        connectiondatamode = False
        weightdatamode = False
        biasdatamode = False

        neuraldatadic = dict()
        layerdatadics = []
        connectiondatadics = []
        weightsdata = []
        biasdata = []

        k_layer = -1
        k_connection = -1
        k_weight = -1
        k_bias = -1
        for line in txt_file:
            tmp_line = line.rstrip()
            if(neuraldatamode):
                if(tmp_line == Network.layersmodestart):
                    neuraldatamode = False
                    layerdatamode = True
                else:
                    tmp = tmp_line.split('\t')
                    neuraldatadic[tmp[0]] = tmp[1:]
            if(layerdatamode):
                if(tmp_line == Network.connectionmodestart):
                    layerdatamode = False
                    connectiondatamode = True
                else:
                    if(tmp_line == Network.layersmodestart):
                        k_layer += 1
                        layerdatadics.append(dict())
                    else:
                        tmp = tmp_line.split('\t')
                        layerdatadics[k_layer][tmp[0]] = tmp[1:]
            if(connectiondatamode):
                if(tmp_line == Network.weightmodestart):
                    connectiondatamode = False
                    weightdatamode = True
                else:
                    if(tmp_line == Network.connectionmodestart):
                        k_connection += 1
                        connectiondatadics.append(dict())
                    else:
                        tmp = tmp_line.split('\t')
                        connectiondatadics[k_connection][tmp[0]] = tmp[1:]
            if(weightdatamode):
                if(tmp_line == Network.biasmodestart):
                    weightdatamode = False
                    biasdatamode = True
                else:
                    if(tmp_line == Network.weightmodestart):
                        k_weight += 1
                        weightsdata.append([])
                    else:
                        weightsdata[k_weight].append(tmp_line)
            if(biasdatamode):
                if(tmp_line == Network.biasmodestart):
                    k_bias += 1
                    biasdata.append([])
                else:
                    biasdata[k_bias].append(tmp_line)
        # Treat Data in Dictionaries
        neural = Network(int(neuraldatadic['input'][0]),int(neuraldatadic['output'][0]))
        neuraldic = dict()
        neuraldic['jitterNoiseLimit'] = float(neuraldatadic['jitterNoiseLimit'][0])
        neuraldic['trainingEpochs'] = int(neuraldatadic['trainingEpochs'][0])
        neuraldic['jitterEpoch'] = int(neuraldatadic['jitterEpoch'][0])
        neuraldic['trainingMethod'] = TrainingMethod[neuraldatadic['trainingMethod'][0]]
        neuraldic['mu_inputs'] = [float(i) for i in neuraldatadic['mu_inputs']]
        neuraldic['sigma_inputs'] = [float(i) for i in neuraldatadic['sigma_inputs']]
        neuraldic['mu_outputs'] = [float(i) for i in neuraldatadic['mu_outputs']]
        neuraldic['sigma_outputs'] = [float(i) for i in neuraldatadic['sigma_outputs']]
        neural.setTrainingArgs(**neuraldic)
        
        layersdic = []
        k_layer = 0
        for layer in layerdatadics:
            layersdic.append(dict())
            layersdic[k_layer]['neurons'] = int(layerdatadics[k_layer]['neurons'][0])
            layersdic[k_layer]['activationFunction'] = getattr(ActivationFunctions,layerdatadics[k_layer]['activationFunction'][0])
            learningrateargs = [float(i) for i in layerdatadics[k_layer]['learningRateFunction'][1:]]
            layersdic[k_layer]['learningRateFunction'] = getattr(LearningRateFunctions,layerdatadics[k_layer]['learningRateFunction'][0])(*learningrateargs)
            biasinitializerargs = [float(i) for i in layerdatadics[k_layer]['biasInitializer'][1:]]
            layersdic[k_layer]['biasInitializer'] = getattr(InitializationFunctions,layerdatadics[k_layer]['biasInitializer'][0])(*biasinitializerargs)
            k_layer += 1
        connectiondic = []
        k_connection = 0
        for connection in connectiondatadics:
            connectiondic.append(dict())
            connectiondic[k_connection]['connectionMode'] = ConnectionMode[connectiondatadics[k_connection]['connectionMode'][0]]
            weightinitializerargs = [float(i) for i in connectiondatadics[k_connection]['weightsInitializer'][1:]]
            connectiondic[k_connection]['weightsInitializer'] = getattr(InitializationFunctions,connectiondatadics[k_connection]['weightsInitializer'][0])(*weightinitializerargs)
            connectiondic[k_connection]['momentum'] = float(connectiondatadics[k_connection]['momentum'][0])
            k_connection += 1

        
        hiddenlayersdata = dict()
        hiddenlayersdata['layersData'] = layersdic
        hiddenlayersdata['connectionData'] = connectiondic
        
        neural.BuildHiddenLayers(**hiddenlayersdata)

        k_layer = 0
        for layer in biasdata:
            k_neuron = 0
            for neuron in neural.layers[k_layer].neurons:
                neuron.bias = float(layer[k_neuron])
                k_neuron += 1
            k_layer += 1

        k_connection = 0
        for connection in weightsdata:
            k_synapse = 0
            for synapse in neural.connectors[k_connection].synapses:
                synapse.weight = float(connection[k_synapse])
                k_synapse += 1
            k_connection += 1
        
        return neural


    def Save(self,txt_file):
        tmp = 'input\t' + str(self.n_input) + '\n'
        tmp += 'output\t' + str(self.m_output) + '\n'
        tmp += 'trainingMethod\t' + str(self.TrainingMethod.name) + '\n'
        tmp += 'jitterNoiseLimit\t' + str(self.jitterNoiseLimit) + '\n'
        tmp += 'jitterEpoch\t' + str(self.jitterEpoch) + '\n'
        tmp += 'trainingEpochs\t' + str(self.trainingEpochs) + '\n'
        tmp += 'normalized\t' + str(self.normalized) + '\n'
        tmp += 'mu_inputs\t'
        counter = 0
        for mu in self.mu_inputs:
            tmp += str(mu)
            if(counter < len(self.mu_inputs) - 1):
                tmp += '\t'
            else:
                tmp += '\n'
            counter+=1
        tmp += 'sigma_inputs\t'
        counter = 0
        for sigma in self.sigma_inputs:
            tmp += str(sigma)
            if(counter < len(self.sigma_inputs) - 1):
                tmp += '\t'
            else:
                tmp += '\n'
            counter+=1
        tmp += 'mu_outputs\t'
        counter = 0
        for mu in self.mu_outputs:
            tmp += str(mu)
            if(counter < len(self.mu_outputs) - 1):
                tmp += '\t'
            else:
                tmp += '\n'
            counter+=1
        tmp += 'sigma_outputs\t'
        counter = 0
        for sigma in self.sigma_outputs:
            tmp += str(sigma)
            if(counter < len(self.sigma_outputs) - 1):
                tmp += '\t'
            else:
                tmp += '\n'
            counter+=1
        for layer in self.hiddenlayers:
            tmp += Network.layersmodestart + '\n'
            tmp += 'neurons\t' + str(layer.neuronCount) + '\n'
            tmp += 'useFixedBiasValues\t' + str(layer.useFixedBiasValues) + '\n'
            tmp += 'activationFunction\t' + str(layer.activationFunction.__name__) + '\n'
            tmp += 'learningRateFunction\t' + str(layer.learningRateFunction) + '\n'
            tmp += 'biasInitializer\t' + str(layer.biasInitializer) + '\n'
        for connection in self.connectors:
            tmp += Network.connectionmodestart + '\n'
            tmp += 'connectionMode\t' + str(connection.connectionMode.name) + '\n'
            tmp += 'weightsInitializer\t' + str(connection.weightInitializer) + '\n'
            tmp += 'momentum\t' + str(connection.momentum) + '\n'
        for connection in self.connectors:
            tmp += Network.weightmodestart + '\n'
            for synapse in connection.synapses:
                tmp += str(synapse.weight) + '\n'
        for layer in self.layers:
            tmp += Network.biasmodestart + '\n'
            for neuron in layer.neurons:
                tmp += str(neuron.bias) + '\n'
        txt_file.write(tmp)



