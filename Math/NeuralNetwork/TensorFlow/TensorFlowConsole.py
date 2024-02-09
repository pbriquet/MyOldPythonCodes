from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import os

class TensorFlowNN:
    def __init__(self,Xdim,Ydim,layersData):
        '''
        :layersData is organized by a list of dictionaries with layers information
        Layer information that are read are:
            'neurons' for neuron count
            'activationFunction' for which function of tensorFlow will be used
        '''
        self.numberOfHiddenLayers = len(layersData)
        self.n = []
        self.activationFunction = []
        self.n.append(Xdim)
        for i in range(self.numberOfHiddenLayers):
            self.n.append(layersData[i]['neurons'])
            self.activationFunction.append(layersData[i]['neurons'])
        self.n.append(Ydim)
        print(self.n)
        self.X = tf.placeholder("float", [None, Xdim])
        self.Y = tf.placeholder("float", [None, Ydim])
        self.Initialize()
    def Initialize(self):
        self.weights = []
        self.biases = []
        self.layers = []
        self.hiddenLayers = []
        for i in range(len(self.n) - 1):
            self.weights.append(tf.Variable(tf.random_normal([self.n[i],self.n[i+1]])))
            self.biases.append(tf.Variable(tf.random_normal([self.n[i+1]])))
        self.layers.append(self.X)
        for i in range(len(self.n) - 2):
            self.hiddenLayers.append(tf.sigmoid(tf.add(tf.matmul(self.layers[i],self.weights[i]), self.biases[i])))
            self.layers.append(self.hiddenLayers[i])
        self.layers.append(tf.sigmoid(tf.add(tf.matmul(self.layers[-1],self.weights[-1]), self.biases[-1])))

    def Training(self,session,XData,YData,learningRate=0.01,trainingEpochs=50000,verbose=True,display_interval=1000):
        self.XData = XData
        self.YData = YData
        self.learningRate = learningRate
        self._trainingSetup()

            # Run the initializer
        session.run(self.init)

        for step in range(1, trainingEpochs+1):
            batch_x = self.XData
            batch_y = self.YData
        # Run optimization op (backprop)
            session.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
            if step % display_interval == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = session.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss))
        print("Optimization Finished!")
        #print(self.prediction.eval(feed_dict={self.X:self.XData}))
        

    def Save(self):
        pass
    def _trainingSetup(self):
        self.prediction = self.logits
        self.loss_op = tf.reduce_mean(tf.square(self.Y - self.logits))
        #self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()

    @property
    def logits(self):
        return self.layers[-1]


def paraboloid(x):
    return (x[0]**2 + x[1]**2)

def generate_points(x_limits,y_limits,n_x,n_y,noise=0.0):
    dx = (x_limits[1] - x_limits[0])/n_x
    dy = (y_limits[1] - y_limits[0])/n_y
    x = []
    z = []
    n = 0
    for i in range(n_x + 1):
        for j in range(n_y + 1):
            x.append([x_limits[0] + i*dx, y_limits[0] + j*dy])
            z.append([paraboloid(x[n]) + np.random.uniform(-1.0,1.0)*noise])
            n += 1
    return x,z

if __name__=='__main__':
    mode = ['data']
    if('data' in mode):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        filepath = os.path.join(__location__,'paraboloid_data.csv')
        XS,Z = generate_points([-2.5,2.5],[-2.5,2.5],15,15,noise=0.0)
        
        X = [i[0] for i in XS]
        Y = [i[1] for i in XS]
        Z = [i[0] for i in Z]

        df = pd.DataFrame({'X':X,'Y':Y,'Z':Z})
        df.to_csv(filepath,sep=';')
    if('build' in mode):
        learning_rate = 0.1
        trainingSteps = 1000
        displayStep = 100
        Ydim = 1
        Xdim = 2

        hiddenLayers = []
        hiddenLayers.append({'neurons':25,'activationFunction':'tanh'})
        hiddenLayers.append({'neurons':25,'activationFunction':'tanh'})

        NN = TensorFlowNN(Xdim,Ydim,hiddenLayers)

        data_x = np.array([[0.0,0.0,1.0],[1.0,-1.0,0.0],[1.0,0.0,1.0],[1.0,1.0,1.0]])
        data_y = np.array([[0.0],[1.0],[1.0],[0.0]])
        #data_x, data_y = generate_points([-2.0,2.0],[-2.0,2.0],15,15)


        with tf.Session() as sess:

            data_x, data_y = generate_points([-0.5,0.5],[-0.5,0.5],5,5,noise=0.01)

            NN.Training(sess,data_x,data_y,trainingEpochs=trainingSteps, learningRate=0.01)

            data_x, data_y = generate_points([-0.5,0.5],[-0.5,0.5],40,40)
            points_x = []
            points_y = []
            points_z = []
            points_z_true = []
            points = []
            for i in range(len(data_y)):
                points_x.append(data_x[i][0])
                points_y.append(data_x[i][1])
                points.append([points_x[i],points_y[i]])
                points_z_true.append(data_y[i])
            value = sess.run(NN.logits, feed_dict={NN.X: points})
            points_z.append(value)
    #i = 0
    #for j in trainingSet.trainingSamples:
    #    print '\nInput is: ' + str(j.inputVector)
    #    print 'Output is: ' + str(neuralNetwork.Run(j.inputVector,verbose=False))
    #    print 'Excepted Output is:' + str(j.outputVector)
    #    i += 1
    if('draw' in mode):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_x,points_y,points_z,c='b')
        ax.scatter(points_x,points_y,points_z_true,c='r')
        plt.show()
