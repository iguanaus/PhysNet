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

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    print("Biases: " , biases, len(biases))
    for i in xrange(0, len(weights)):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"b_"+str(i)+".txt",bias_i,delimiter=',')
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    #Fine if everything is 2D
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
        print("weights:")
        print(weights)
        num_layers = len(weights)-1
        weight_1 = np.loadtxt(output_folder+weight_load_name+"w_"+str(num_layers)+".txt",delimiter=',')
        weight_new_1 = tf.Variable(weight_1,dtype=tf.float32)
        weights[-1] = weight_new_1

        bias_1 = np.loadtxt(output_folder+weight_load_name+"b_"+str(num_layers)+".txt",delimiter=',')
        print("bias: " , bias_1)
        bias_new_1 = tf.Variable(bias_1,dtype=tf.float32)
        biases[-1] = bias_new_1


        #print(weights)
    print("biases: " , len(biases))
    print("weights: " , len(weights))


    return weights , biases

def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    #I will preset the architecture.
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(X,weights[0])),biases[0])#Go into first 10 layer
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[1])),biases[1]) #Second 10
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[2])),biases[2])# Now 5
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[3])),biases[3]) #Now third 10
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[4])),biases[4]) #Now last layer
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

        x0 = t
        x1 = y0 + v*t+-0.5*g*t*t
        x2 = y0 + v*(t+dt)+-0.5*g*(t+dt)*(t+dt)
        x3 = y0 + v*(t+2*dt)+-0.5*g*(t+2*dt)*(t+2*dt)
        x4 = y0 + v*(t+3*dt)+-0.5*g*(t+3*dt)*(t+3*dt)
        x5 = y0 + v*(t+4*dt)+-0.5*g*(t+4*dt)*(t+4*dt)
        x6 = y0 + v*(t+5*dt)+-0.5*g*(t+5*dt)*(t+5*dt)
        x7 = y0 + v*(t+6*dt)+-0.5*g*(t+6*dt)*(t+6*dt)
        x8 = y0 + v*(t+7*dt)+-0.5*g*(t+7*dt)*(t+7*dt)
        x9 = y0 + v*(t+8*dt)+-0.5*g*(t+8*dt)*(t+8*dt)
        x10 = y0 + v*(t+9*dt)+-0.5*g*(t+9*dt)*(t+9*dt)
        x11 = y0 + v*(t+10*dt)+-0.5*g*(t+10*dt)*(t+10*dt)
        x12 = y0 + v*(t+11*dt)+-0.5*g*(t+11*dt)*(t+11*dt)
        x20 = y0 + v*(t+19*dt)+-0.5*g*(t+19*dt)*(t+19*dt)


        x[i] = [x1,x2,x3]#,x4,x5,x6,x7,x8,x9,x10,x11,x12]
        y[i] = [x4]#,x9,x20]


        #randNum1 = random.uniform(min_val, max_val)
        #randNum2 = randNum1#random.uniform(min_val, max_val)
        #x[i] = [randNum1,randNum2]
        #y[i] = [randNum1*randNum2]
    maxNum = 200.0
    print(x)
    print(x.shape)
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            x[i][j] = int(x[i][j]*maxNum)+500.0
    for i in xrange(y.shape[0]):
        y[i] = int(y[i]*maxNum)+500.0

    train_X = x
    train_Y = y
    print x
    print y
    X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    return X_train, y_train, X_val, y_val



def main(output_folder,weight_name_load,spect_to_sample,sample_val,num_layers,n_hidden,percent_val):

    if not os.path.exists(output_folder):
        print("ERROR THERE IS NO OUTPUT FOLDER. PLEASE SYNC THIS TO PART 1")

    train_X, train_Y , val_X, val_Y = get_data(percentTest=percent_val)

    x_size = 3#train_X.shape[1]
    y_size = 1

    # Symbols
    x_pre = tf.placeholder("int32", shape=[None, x_size])
    y = tf.placeholder("int64", shape=[None])

    embedding = tf.get_variable("embedding", [1000, 1000])
    
    inputs = tf.nn.embedding_lookup(embedding, x_pre)

    print("Inputs: " , inputs)

    #Lastly we need to i guess flatten this. I don't know what other way to do it.
    X = tf.reshape(inputs,[-1,3000])
    x_size = 3000

    weights, biases = load_weights(output_folder,weight_name_load,num_layers)

    yhat    = forwardprop(X, weights,biases,num_layers)
    
    # Backward propagation
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yhat,labels=y))
    our_val = tf.argmax(yhat, 1)

    
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        start_time=time.time()
        print("========                         Iterations started                  ========")
        x_set = None
        y_set = None
        if sample_val:
            x_set = val_X
            y_set = val_Y
        else:
            x_set = train_X
            y_set = train_Y
        print("x_set: " , x_set)

        batch_x = x_set[spect_to_sample : (spect_to_sample+1) ]
        batch_y = y_set[spect_to_sample : (spect_to_sample+1) ]
        print("Batch x1: " , batch_x)

        #Now we should step through
        #x_1 = batch_x[0][0]
        #x_2 = batch_x[0][1]
        #x_3 = batch_x[0][2]
        #x_4 = batch_y[0][0]
        #np.reshape(np.loadtxt(output_folder+weight_load_name+"w_"+str(num_layers)+".txt",delimiter=','),(-1,1))

        #yValsToPlot=[]


        #for i in xrange(0,10):
        #batch_x = np.reshape(np.array([x_1,x_2,x_3]),(1,-1))
        #batch_y = np.reshape(np.array([x_4]),(1,-1))
        print("Batch x2 : " , batch_x)
        mycost = sess.run(cost,feed_dict={x_pre:batch_x,y:batch_y.reshape([-1])})
        myvals0 = sess.run(our_val,feed_dict={x_pre:batch_x,y:batch_y.reshape([-1])})
        #x_1 = x_2
        #x_2 = x_3
        #x_4 = myvals0[0][0]
        #x_3 = x_4
        #print("my Vals: " , myvals0)
        #x_4 = myvals0[0][0]
        #yValsToPlot.append(x_4)
        #yValsToPlot.append(myvals0[0][1])
        #yValsToPlot.append(myvals0[0][2])
        or_val = batch_y
        yValsToPlot = myvals0


        filename = output_folder + "test_out_file_single_"+str(spect_to_sample) + ".txt"
        f = open(filename,'w')
        f.write("XValue\nActual\nPredicted\n")
        print("Batch: " , batch_x)
        for i in xrange(0,len(batch_x[0])):
            item = batch_x[0][i]
            if (i != len(batch_x[0])-1):
                f.write(str(item)+",")
            else:
                f.write(str(item))
        f.write("\n")
        #f.write(str(batch_x[0][0])+","+str(batch_x[0][1])+'\n')
        #f.write(str(batch_x[0])+"\n")
        for i in xrange(0, len(or_val)):
            item = or_val[i]
            if (i != len(or_val)-1):
                f.write(str(item)+",")
            else:
                f.write(str(item))

        f.write("\n")

        for i in xrange(0,len(yValsToPlot)):
            item = yValsToPlot[i]
            if (i != len(yValsToPlot)-1):
                f.write(str(item)+",")
            else:
                f.write(str(item))
        f.flush()
        f.close()
        print("Cost: " , mycost)
        print(myvals0)
        print("Wrote to: " + str(filename))

    print "========Writing completed in : " + str(time.time()-start_time) + " ========"
        
    sess.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    #Use Project 4
    parser.add_argument("--output_folder",type=str,default='results/Project_9_size_20/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--spect_to_sample",type=int,default=22) #Zero Indexing
    parser.add_argument("--sample_val",type=str,default="True")
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=100)
    parser.add_argument("--percent_val",default=0.2)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'output_folder':dict['output_folder'],
            'weight_name_load':dict['weight_name_load'],
            'spect_to_sample':dict['spect_to_sample'],
            'sample_val':dict['sample_val'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val']
            }

    main(**kwargs)



