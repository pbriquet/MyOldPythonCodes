import matplotlib.pyplot as plt
import seaborn as sns

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from keras import backend as K
from keras.layers import Dense
from sklearn.cross_validation import train_test_split

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

class MixtureDensityNetwork:
    def __init__(self,k):
        self.k = k
    def mapping(self,x):
        hidden1 = Dense(15,activation='relu')(x)
        hidden2 = Dense(15,activation='relu')(hidden1)
        self.mus = Dense(self.k)(hidden2)
        self.sigmas = Dense(self.k,activation=K.exp)(hidden2)
        self.pi = Dense(self.k,activation=K.softmax)(hidden2)
    def log_prob(self,xs,zs = None):
        x,y = xs
        self.mapping(x)
        result = tf.exp(Normal.logpdf(y,self.mus,self.sigmas))
        result = tf.mul(result,self.pi)
        result = tf.reduce_sum(result,1)
        result = tf.log(result)
        return tf.reduce_sum(result)

ed.set_seed(42)
model = MixtureDensityNetwork(20)

X = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))

inference = ed.MAP(model,[X, y])
sess = tf.Session()
K.set_session(sess)
inference.initilize(sess=sess)