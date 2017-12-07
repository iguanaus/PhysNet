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

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=.1)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    print("Biases: " , biases, len(biases))
    for i in xrange(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
        if i != num_layers:
            for j in xrange(0,5):
                bias_i = biases[5*i+j].eval()
                np.savetxt(output_folder+weight_name_save+"b_"+str(i)+"-"+str(j)+".txt",bias_i,delimiter=',')
        else:
            bias_i = biases[5*i].eval()
            np.savetxt(output_folder+weight_name_save+"b_"+str(i)+".txt",bias_i,delimiter=',')
        print("Bias: " , i, " : ", bias_i)
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    #Fine if everything is 2D
    for i in xrange(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        if i != num_layers:
            for j in xrange(0,5):
                bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+"-"+str(j)+".txt",delimiter=',')
                b_i = tf.Variable(bias_i,dtype=tf.float32)
                biases.append(b_i)
        else:
            bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+".txt",delimiter=',')
            b_i = tf.Variable(bias_i,dtype=tf.float32)
            biases.append(b_i)

        print("Last Bias: " , bias_i)
    #If it needs to be one D
    if True:
        print("weights:")
        print(weights)
        weight_1 = np.reshape(np.loadtxt(output_folder+weight_load_name+"w_"+str(num_layers)+".txt",delimiter=','),(-1,1))
        weight_new_1 = tf.Variable(weight_1,dtype=tf.float32)
        weights[-1] = weight_new_1

        bias_1 = np.reshape(np.loadtxt(output_folder+weight_load_name+"b_"+str(num_layers)+".txt",delimiter=','),(-1,1))
        print("bias: " , bias_1)
        bias_new_1 = tf.Variable(bias_1,dtype=tf.float32)
        biases[-1] = bias_new_1


        #print(weights)
    print("biases: " , len(biases))
    print("weights: " , len(weights))


    return weights , biases


def fourier_expand(x,coefs):
    firstterm = tf.multiply(tf.constant(1.0),coefs[0])
    secondterm = tf.multiply(x,coefs[1])
    thirdterm = tf.multiply(tf.multiply(x,x),coefs[2])
    fourthterm = tf.multiply(tf.multiply(tf.multiply(x,x),x),coefs[3])
    return firstterm+secondterm+thirdterm+fourthterm



def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    for i in xrange(0, num_layers):
        coeffs = [biases[i*5+1],biases[i*5+2],biases[i*5+3],biases[i*5+4]]
        if i ==0:
            htemp = tf.add(fourier_expand(tf.matmul(X,weights[i]),coeffs),biases[i*5])
            #htemp = tf.add(tf.nn.sigmoid(tf.matmul(X,weights[i])),biases[i])    
        else:   
            htemp = tf.add(fourier_expand(tf.matmul(htemp,weights[i]),coeffs),biases[i*5])
            #htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[i])),biases[i])
        print("Bias: " , i, " : ", biases[i])
    #drop_out = tf.nn.dropout(htemp,0.9)
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    print("Last bias: " , biases[-1])
    return (yval)

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(percentTest=.2,random_state=42):
    #Gen the x data
    #Gen the y data
    num_entries = 100000
    min_val = 0.0
    max_val = 100.0
    x = np.zeros(shape=(num_entries,2))
    y = np.zeros(shape=(num_entries,1))
    for i in xrange(0,num_entries):
        randNum1 = random.uniform(min_val, max_val)
        randNum2 = randNum1#random.uniform(min_val, max_val)
        x[i] = [randNum1,randNum2]
        y[i] = [randNum1*randNum2]
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

    x_size = train_X.shape[1]
    y_size = train_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []
    biases = []
    coef = []
    # Weight initializations
    if reuse_weights:
        (weights, biases) = load_weights(output_folder,weight_name_load,num_layers)

    else:
        for i in xrange(0,num_layers):
            if i ==0:
                weights.append(init_weights((x_size,n_hidden)))
            else:
                weights.append(init_weights((n_hidden,n_hidden)))
            biases.append(init_bias(n_hidden))
            biases.append(init_bias(n_hidden))
            biases.append(init_bias(n_hidden))
            biases.append(init_bias(n_hidden))
            biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,y_size)))
        biases.append(init_bias(y_size))
        #biases.append(init_bias(y_size))
        #biases.append(init_bias(y_size))
        #biases.append(init_bias(y_size))
        #biases.append(init_bias(y_size))

    #for ele in coef:
    #    biases.append(ele)
    print("Coefficients")
    print(biases)

    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))

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
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            #print(batch_x)
            #print(batch_y)
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            myvals, new_loss = sess.run([yhat,cost],feed_dict={X:batch_x,y:batch_y})
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
                    val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
                    print("Validation loss: " , str(val_loss))
                    val_loss_file.write(str(float(val_loss))+str("\n"))
                    val_loss_file.flush()

                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                cum_loss = 0
        save_weights(weights,biases,output_folder,weight_name_save,num_layers)
    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--reuse_weights",type=str,default='False')
    parser.add_argument("--output_folder",type=str,default='results/10th_try/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=1000)
    parser.add_argument("--numEpochs",type=int,default=50)
    parser.add_argument("--lr_rate",default=.005)
    parser.add_argument("--lr_decay",default=.9999)
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




