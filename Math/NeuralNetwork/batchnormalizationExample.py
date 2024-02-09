import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from math import floor, ceil
import sklearn.model_selection as sk


def generate_points(function,x_limits,y_limits,n_x,n_y,noise=0.0):
    dx = (x_limits[1] - x_limits[0])/n_x
    dy = (y_limits[1] - y_limits[0])/n_y
    x = []
    y = []
    z = []
    n = 0
    for i in range(n_x + 1):
        for j in range(n_y + 1):
            x.append(x_limits[0] + i*dx)
            y.append(y_limits[0] + j*dy)
            z.append(function(x[-1],y[-1]) + np.random.uniform(-1.0,1.0)*noise)
            n += 1
    return np.array(x),np.array(y),np.array(z)

if __name__=='__main__':
    mode = ['train','draw']
    verbose = True
    func = lambda x,y: np.exp(-0.1*x**2) + np.arctan(np.sqrt(np.abs(y)))
    #func = lambda x,y: x**2 + y**2
    xlim = [-1.0,2.0]
    ylim = [-3.0,2.0]
    n_x = 30
    n_y = 30
    X,Y,Z = generate_points(func,xlim,ylim,n_x,n_y)

    df = pd.DataFrame({'X':X,'Y':Y,'Z':Z})
    test_size = 0.0
    learningRate = 0.01
    x_batch = np.transpose(np.array([df['X'],df['Y']]))
    z_batch = np.transpose(df['Z'])
    
    # Separate batch into Train Set and Test Set. Test set is hidden from training, but checks accuracy
    X_train, X_test, Z_train, Z_test = sk.train_test_split(x_batch,z_batch,test_size=test_size,random_state=42)
    
    dim_input = 2
    dim_output = 1
    neurons0 = 3
    neurons1 = 4
    epsilon = 1e-3
    trainingEpochs = 3000
    display_interval = 100

    X_tf = tf.placeholder(tf.float32,shape=[None,dim_input],name='X')
    Z_tf = tf.placeholder(tf.float32,shape=[None,dim_output],name='Y')
    
    batch_mean_i, batch_var_i = tf.nn.moments(X_tf,[0])
    scale_i = tf.Variable(tf.ones([dim_input]),name='scale_i',trainable=True)
    beta_i = tf.Variable(tf.zeros([dim_input]),name='beta_i',trainable=True)
    BN_i = tf.nn.batch_normalization(X_tf,batch_mean_i,batch_var_i,beta_i,scale_i,epsilon,name='BN_i')

    if(verbose):
        print('Input')
        print('\tbatch_mean : ' + str(batch_mean_i))
        print('\tbatch_var : ' + str(batch_var_i))
        print('\tgammas : ' + str(scale_i))
        print('\tbetas : ' + str(beta_i))
        print('\tBN : ' + str(BN_i))

    W_0 = tf.Variable(tf.random_normal([dim_input,neurons0]), trainable=True,name='W_0')
    b_0 = tf.Variable(tf.random_normal([neurons0]), trainable=True,name='b_0')
    arg0 = tf.add(tf.matmul(BN_i,W_0),b_0)
    batch_mean0, batch_var0 = tf.nn.moments(arg0,[0])
    scale0 = tf.Variable(tf.ones([neurons0]),name='scale0',trainable=True)
    beta0 = tf.Variable(tf.zeros([neurons0]),name='beta0',trainable=True)
    BN0 = tf.nn.batch_normalization(arg0,batch_mean0,batch_var0,beta0,scale0,epsilon,name='BN0')
    act0 = tf.nn.tanh
    layer_0 = act0(BN0,name='layer0')

    if(verbose):
        print('0')
        print('\tweight : ' + str(W_0))
        print('\tbias : ' + str(b_0))
        print('\targs : ' + str(arg0))
        print('\tact : ' + str(act0))
        print('\tbatch_mean : ' + str(batch_mean0))
        print('\tbatch_var : ' + str(batch_var0))
        print('\tgammas : ' + str(scale0))
        print('\tbetas : ' + str(beta0))
        print('\tBN : ' + str(BN0))
        print('\tlayer : ' + str(layer_0))

    W_1 = tf.Variable(tf.random_normal([neurons0,neurons1]), trainable=True,name='W_1')
    b_1 = tf.Variable(tf.random_normal([neurons1]), trainable=True,name='b_1')
    arg1 = tf.add(tf.matmul(layer_0,W_1),b_1)
    batch_mean1, batch_var1 = tf.nn.moments(arg1,[0])
    scale1 = tf.Variable(tf.ones([neurons1]),name='scale1',trainable=True)
    beta1 = tf.Variable(tf.zeros([neurons1]),name='beta1',trainable=True)
    BN1 = tf.nn.batch_normalization(arg1,batch_mean1,batch_var1,beta1,scale1,epsilon,name='BN1')
    act1 = tf.nn.tanh
    layer_1 = tf.nn.tanh(BN1,name='layer1')
    
    if(verbose):
        print('1')
        print('\tweight : ' + str(W_1))
        print('\tbias : ' + str(b_1))
        print('\targs : ' + str(arg1))
        print('\tact : ' + str(act1))
        print('\tbatch_mean : ' + str(batch_mean1))
        print('\tbatch_var : ' + str(batch_var1))
        print('\tgammas : ' + str(scale1))
        print('\tbetas : ' + str(beta1))
        print('\tBN : ' + str(BN1))
        print('\tlayer : ' + str(layer_1))

    W_2 = tf.Variable(tf.random_normal([neurons1,dim_output]), trainable=True,name='W_2')
    b_2 = tf.Variable(tf.random_normal([dim_output]), trainable=True,name='b_2')
    arg2 = tf.add(tf.matmul(layer_1,W_2),b_2)
    batch_mean2, batch_var2 = tf.nn.moments(arg2,[0])
    scale2 = tf.Variable(tf.ones([dim_output]),name='scale2',trainable=True)
    beta2 = tf.Variable(tf.zeros([dim_output]),name='beta2',trainable=True)
    BN2 = tf.nn.batch_normalization(arg2,batch_mean2,batch_var2,beta2,scale2,epsilon,name='BN2')
    act2 = tf.identity
    layer_2 = act2(BN2,name='layer2')

    if(verbose):
        print('2')
        print('\tweight : ' + str(W_2))
        print('\tbias : ' + str(b_2))
        print('\targs : ' + str(arg2))
        print('\tbatch_mean : ' + str(batch_mean2))
        print('\tbatch_var : ' + str(batch_var2))
        print('\tact : ' + str(act2))
        print('\tgammas : ' + str(scale2))
        print('\tbetas : ' + str(beta2))
        print('\tBN : ' + str(BN2))
        print('\tlayer : ' + str(layer_2))
    
    batch_mean_o, batch_var_o = tf.nn.moments(Z_tf,[0])
    scale_o = tf.Variable(tf.ones([dim_output]),name='scale_o',trainable=True)
    beta_o = tf.Variable(tf.zeros([dim_output]),name='beta_o',trainable=True)
    BN_o = tf.nn.batch_normalization(Z_tf,batch_mean_o,batch_var_o,beta_o,scale_o,epsilon,name='BN_o')
 
    logits = scale_o*layer_2 + beta_o

    if(verbose):
        print('Output')
        print('\tbatch_mean : ' + str(batch_mean_o))
        print('\tbatch_var : ' + str(batch_var_o))
        print('\tgammas : ' + str(scale_o))
        print('\tbetas : ' + str(beta_o))
        print('\tBN : ' + str(BN_o))
        print('\tlogits : ' + str(logits))
    
    loss_op = tf.reduce_mean(tf.pow(logits - Z_tf,2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train_op = optimizer.minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Z_tf, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        #print(sess.run(batch_mean0,feed_dict={Full_X:x_batch}))
        #print(sess.run(batch_var0,feed_dict={Full_X:x_batch}))
        
        if('train' in mode):
            for step in range(1, trainingEpochs+1):
            # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X_tf: X_train,Z_tf: np.transpose([Z_train])})
                if step % display_interval == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X_tf: X_train,
                                                                        Z_tf: np.transpose([Z_train])})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4e}".format(loss))
                    print(sess.run(scale_o,feed_dict={X_tf:x_batch}))
            print("Optimization Finished!")
        if('draw' in mode):
            X,Y,Z = generate_points(func,xlim,ylim,30,30)

            df = pd.DataFrame({'X':X,'Y':Y,'Z':Z})
            test_size = 0.0
            learningRate = 0.01
            x_batch = np.transpose(np.array([df['X'],df['Y']]))
            z_batch = np.transpose(df['Z'])
            
            # Separate batch into Train Set and Test Set. Test set is hidden from training, but checks accuracy
            X_train, X_test, Z_train, Z_test = sk.train_test_split(x_batch,z_batch,test_size=test_size,random_state=42)

            Z_NN = sess.run(logits,feed_dict={X_tf:x_batch})
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(X,Y,Z,cmap=cm.hot)
            ax.scatter(X,Y,Z_NN,cmap=cm.hot)
            plt.show()
