from __future__ import absolute_import
from __future__ import division
from __future__    import print_function


from numpy import genfromtxt
import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import *
 
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
import pandas as pd
import re

skipAmount = 10.0 #This is how frequent the data should be sampled.
emaPredictInFuture = 50 #This is how much into the future it should have to predict
seq_len = 1 #Sequence length.
smoothedPrice = 200 #This just smooths the input. 
reuse = True
  
#THis needs to take in the data, then return the data in a list of
#The output should be in a np array form. Note that the y value doesn't have to be returned. 

def file_data(filename="data/waveEqSim_A.csv"):
    #filename="data/06_01_10.csv"
    my_data = genfromtxt(filename, delimiter=',')
    return my_data



def main():
    # --- Set data params ----------------
    #Create Data
    max_len_data = 1000000000

    #data , standardDev,meanVal= file_data()
    data2 = file_data()

    n_input = len(data2[0])
    print("N input length: " , n_input)

    n_output = n_input
    n_hidden = 100
    learning_rate = 0.001
    decay = 0.9
    numEpochs = 200
    #reuse = False

    #Structure of this will be [weekday,seconds*1000,intPrice,volume]

    X = tf.placeholder("float32",[None,seq_len,n_input])
    Y = tf.placeholder("float32",[None,seq_len,n_input])

    # Input to hidden layer
    cell = None
    h = None
    num_layers = 3
    #h_b = None
    sequence_length = [seq_len] * 1

    cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)

    #cells = rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    cells = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias =1) for _ in range(num_layers)],state_is_tuple=True)

    if h == None:
        h = cells.zero_state(1,tf.float32)

    hidden_out, states = tf.nn.dynamic_rnn(cells, X, sequence_length=sequence_length, dtype=tf.float32,initial_state=h)


    # Hidden Layer to Output
    V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
            dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_output], \
            dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    hidden_out_list = tf.unstack(hidden_out, axis=1)
    temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
    output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias)


    # define evaluate process
    print("Output data: " , output_data)
    print("Labels: " , Y)

    cost = tf.reduce_sum(tf.square(Y-output_data))

    #cost = tf.reduce_sum(tf.square(output_data-Y))
    #correct_pred = tf.equal(tf.round(output_data*standardDev+meanVal), tf.round(Y*standardDev+meanVal))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # --- Initialization ----------------------
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           5000, 0.9, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
    init = tf.global_variables_initializer()

    for i in tf.global_variables():
        print(i.name)

    step = 0
    savename="models/"
    if not os.path.exists(savename):
        os.makedirs(savename)
    saver = tf.train.Saver()
    numFile = 0


    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
        print("Session Created")
        sess.run(init)
        if reuse:
            saver.restore(sess, os.path.join(savename,'my_model'))


            #new_saver = tf.train.import_meta_graph("modelFile"+'.meta')
            #new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        steps = []
        losses = []
        accs = []
        validation_losses = []
        curEpoch = 0

        
        training_state = None
        i = 0
        print ("Number train: " , len(data2))
        train_file_name = "loss_val.csv"
        train_loss_file = open(train_file_name,'w')
        
        outputList = np.array([[]])
        desiredList = np.array([[]])
        #In this, we pass in the first then it just goes for it. 
        i+= 1
        myTrain_x = data2[seq_len*i:seq_len*(i+1)].reshape((1,seq_len,n_input))
        out_data = myTrain_x


        while i <= int(((len(data2)-1.0)/seq_len)-2):
            
            print("I: " , i)
            #myTrain_x = data2[seq_len*i:seq_len*(i+1)].reshape((1,seq_len,n_input))
            myTrain_x = out_data
            
            myTrain_y = data2[seq_len*i+1:seq_len*(i+1)+1].reshape((1,seq_len,n_input))
            i += 1
            #print("X Predict: " , myTrain_x)
            myfeed_dict={X: myTrain_x, Y: myTrain_y}
            if training_state is not None:
                myfeed_dict[h] = training_state
            
            out_data,loss,training_state,output_data_2 = sess.run([output_data,cost,states,output_data], feed_dict = myfeed_dict)
            
            print("Y Values: " , myTrain_y[0])
            
            print("Output data: " , out_data[0])
            
            print("Epoch: " + str(curEpoch) + " Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
            outputVal = np.array(out_data[0])
            correctVal = np.array(myTrain_y[0])
            print("OutputVal: " , outputVal)
            #outputVal = np.array(output_data_2*standardDev+meanVal)
            #correctVal = myTrain_y*standardDev+meanVal
            #Okay we need to write two columns to the file, one for outputVal, one for correctVal

            #print("Output: " , outputVal[0])
            print(len(outputList))

            if len(outputList[0]) <= 0:
                outputList = outputVal
                print("OutputList within the if: " , outputList)
                desiredList = correctVal
            else:
                print("Jumping to this....")
                print("outputList: ")
                print(outputList)
                print("OutputVal: " )
                print(outputVal)
                outputList = np.append(outputList,outputVal,axis=0)
                desiredList = np.append(desiredList,correctVal,axis=0)
            #print(outputList)
            #print(desiredList)
            print("output list complete: " )
            print(outputList)
            
            train_loss_file.write(str(loss)+"\n")
            train_loss_file.flush()

            #break

            #print("My train: " , correctVal)
            #print("Output - myTrain: " , outputVal-correctVal)
        print ("Final Output list: ")
        print(outputList)
        print("Final Desired List: ")
        print(desiredList)

        np.savetxt('y_predictList_vol500_price_long.csv', outputList, delimiter=',')
        np.savetxt('y_actualList_vol500_price_long.csv', desiredList, delimiter=',')

        train_loss_file.close()

        print("Optimization Finished!")
        


if __name__=="__main__":
    
    main()
