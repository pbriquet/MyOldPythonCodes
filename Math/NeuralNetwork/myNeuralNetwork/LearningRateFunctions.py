class LearningRateFunctions:
    class LinearFunction:
        def __init__(self,*args):
            self.initialLearningRate = args[0]
            self.finalLearningRate = args[1]
        def GetLearningRate(self,currentIteration,trainingEpochs):
            return self.initialLearningRate + (self.finalLearningRate - self.initialLearningRate) * currentIteration / trainingEpochs
        def __str__(self):
            tmp = 'LinearFunction\t'
            tmp += str(self.initialLearningRate) + '\t'
            tmp += str(self.finalLearningRate)
            return tmp
    class HyperbolicFunction:
        def __init__(self,*args):
            self.initialLearningRate = args[0]
            self.finalLearningRate = args[1]
        def GetLearningRate(self,currentIteration,trainingEpochs):
            return self.initialLearningRate + (self.finalLearningRate - self.initialLearningRate) * currentIteration / trainingEpochs
        def __str__(self):
            tmp = 'HyperbolicFunction\t'
            tmp += str(self.initialLearningRate) + '\t'
            tmp += str(self.finalLearningRate)
            return tmp
    class ExponentialFunction:
        def __init__(self,*args):
            pass

