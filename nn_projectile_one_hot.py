'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radi of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights. 
'''

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os
import random

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

num_decay = 4320000

num_entries = 100000

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=.1)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers,embedding):
    print("Biases: " , biases, len(biases))
    for i in xrange(0, len(weights)):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"b_"+str(i)+".txt",bias_i,delimiter=',')
    np.savetxt(output_folder+weight_name_save+"e_0.txt",embedding.eval(),delimiter=',')

    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    #Fine if everything is 2D
    try:
        embedding = np.loadtxt(output_folder+weight_load_name+"e_0.txt",delimiter=',')
        embedding = tf.Variable(embedding,dtype=tf.float32)
    except:
        pass
    for i in xrange(0,1000):
        try:
            weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
            w_i = tf.Variable(weight_i,dtype=tf.float32)
            weights.append(w_i)
            bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+".txt",delimiter=',')
            b_i = tf.Variable(bias_i,dtype=tf.float32)
            biases.append(b_i)
        except:
            break
    #If it needs to be one D
    if True:
        #print("weights:")
        #print(weights)
        num_layers = len(weights)-1
        weight_1 = np.loadtxt(output_folder+weight_load_name+"w_"+str(num_layers)+".txt",delimiter=',')
        weight_new_1 = tf.Variable(weight_1,dtype=tf.float32)
        weights[-1] = weight_new_1

        bias_1 = np.loadtxt(output_folder+weight_load_name+"b_"+str(num_layers)+".txt",delimiter=',')
        #print("bias: " , bias_1)
        bias_new_1 = tf.Variable(bias_1,dtype=tf.float32)
        biases[-1] = bias_new_1


        #print(weights)
    #print("biases: " , len(biases))
    #print("weights: " , len(weights))


    return weights , biases, embedding



def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    #I will preset the architecture.
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(X,weights[0])),biases[0])#Go into first 10 layer
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[1])),biases[1]) #Second 10
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[2])),biases[2])# Now 5
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[3])),biases[3]) #Now third 10
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[4])),biases[4]) #Now last layer
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    #print("Last bias: " , biases[-1])
    return (yval)

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(percentTest=.2,random_state=42):
    #Gen the x data
    #Gen the y data
    
    #g = -.2394
    #min_val = 0.0
    #max_val = 100.0
    

    x = np.zeros(shape=(num_entries,3))
    y = np.zeros(shape=(num_entries,1))
    for i in xrange(0,num_entries):
        #I will feed in two timesteps of the data
        y0 = random.uniform(0.0,.5)
        v = random.uniform(0.0,.5)# These should be random. 
        g = random.uniform(0.0,.5)
        t = random.uniform(0.0,.5)
        dt = random.uniform(0.0,.5)

        x1 = y0 + v*t+-0.5*g*t*t
        x2 = y0 + v*(t+dt)+-0.5*g*(t+dt)*(t+dt)
        x3 = y0 + v*(t+2*dt)+-0.5*g*(t+2*dt)*(t+2*dt)
        x4 = y0 + v*(t+3*dt)+-0.5*g*(t+3*dt)*(t+3*dt)


        x_temp = [x1,x2,x3]
        y_temp = [x4]
        if (min(x_temp) > 0.0 and min(y_temp) > 0.0 and max(x_temp) < 1.0 and max(y_temp) < 1.0):
            x[i] = [x1,x2,x3]
            y[i] = [x4]

    #Transform it to integers from 0-1000.
    maxNum = 100.0
    print(x)
    print(x.shape)
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            x[i][j] = int(x[i][j]*maxNum)#+500.0
    for i in xrange(y.shape[0]):
        y[i] = int(y[i]*maxNum)#+500.0

    train_X = x
    train_Y = y
    print x
    print y
    X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    return X_train, y_train, X_val, y_val

def main(reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_X, train_Y , val_X, val_Y = get_data(percentTest=percent_val)

    x_size = 3#train_X.shape[1]
    y_size = 1#train_Y.shape[1]
    print("X Size: " , x_size)

    # Symbols
    x_pre = tf.placeholder("int32", shape=[None, x_size])
    y = tf.placeholder("int64", shape=[None])

    n_hidden = 50

    x_size = 300

    weights = []
    biases = []
    coef = []
    embedding = None
    # Weight initializations
    if reuse_weights:
        (weights, biases,embedding) = load_weights(output_folder,weight_name_load,num_layers)

    else:
        embedding = tf.get_variable("embedding", [100, 100])
        
        weights.append(init_weights((x_size,n_hidden))) #First
        biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,n_hidden))) #Second
        biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,5)))   #Mid 5
        biases.append(init_bias(5))
        weights.append(init_weights((5,n_hidden))) #Third
        biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,n_hidden))) #Fourth
        biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,100))) #Out
        biases.append(init_bias(100))

    #for ele in coef:
    #    biases.append(ele)
    print("Coefficients")
    print(biases)

    inputs = tf.nn.embedding_lookup(embedding, x_pre)

    print("Inputs: " , inputs)

    #Lastly we need to i guess flatten this. I don't know what other way to do it.
    X = tf.reshape(inputs,[-1,300])



    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)

    print("yHat: " , yhat)
    
    # Backward propagation

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yhat,labels=y))
    correct_prediction = tf.equal(tf.argmax(yhat, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #correct
    #cost = tf.reduce_sum(tf.square(y-yhat))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = lr_rate
    #learning_rate = tf.train.exponential_decay(lr_rate,global_step,num_decay,.96,staircase=False)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=lr_decay).minimize(cost,global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        curEpoch=0
        cum_loss = 0
        numFile = 0 
        while True:
            train_file_name = output_folder+"train_train_loss_" + str(numFile) + ".txt"
            if os.path.isfile(train_file_name):
                numFile += 1
            else:
                break
        train_loss_file = open(train_file_name,'w')
        val_loss_file = open(output_folder+"train_val_loss_"+str(numFile) + "_val.txt",'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while curEpoch < numEpochs:
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch].reshape([-1])
            #print(batch_x.shape)
            #print(batch_y.shape)
            #print(batch_x)
            #print(batch_y)
            sess.run(optimizer, feed_dict={x_pre: batch_x, y: batch_y})
            myvals, new_loss = sess.run([yhat,cost],feed_dict={x_pre:batch_x,y:batch_y})
            #print((myvals[0]-batch_y[0])/myvals[0]*100.0)
            #print('\n')
            #print(myvals[0])
            #print(batch_y[0])
            #print((myvals - batch_y)[0][0])
            cum_loss += new_loss
            step += 1
            if step == int(train_X.shape[0]/n_batch): #Epoch finished
                step = 0
                curEpoch +=1            
                train_loss_file.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss
                    acc , val_loss = sess.run((accuracy,cost),feed_dict={x_pre:val_X,y:val_Y.reshape([-1])})
                    print("Validation loss: " , str(val_loss), str(acc))
                    val_loss_file.write(str(float(val_loss))+str("\n"))
                    val_loss_file.flush()

                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                cum_loss = 0
        save_weights(weights,biases,output_folder,weight_name_save,num_layers,embedding)
    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--reuse_weights",type=str,default='False')
    parser.add_argument("--output_folder",type=str,default='results/Project_12_00_acc/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=10)
    parser.add_argument("--numEpochs",type=int,default=1)
    parser.add_argument("--lr_rate",default=.0000001)
    parser.add_argument("--lr_decay",default=.9)
    parser.add_argument("--num_layers",default=1)
    parser.add_argument("--n_hidden",default=2)
    parser.add_argument("--percent_val",default=.2)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'n_batch':dict['n_batch'],
            'numEpochs':dict['numEpochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val']
            }

    main(**kwargs)




