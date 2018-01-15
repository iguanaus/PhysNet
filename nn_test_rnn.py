#This is going to be simply a really basic RNN that will predict in the waveEqSim_A

from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

from numpy import genfromtxt
import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import *
import matplotlib.pyplot as plt
#from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
import pandas as pd
import re

seq_len = 2 #Sequence length.
  
#THis needs to take in the data, then return the data in a list of
#The output should be in a np array form. Note that the y value doesn't have to be returned. 

def file_data(filename="data/waveEqSim_A.csv"):
	#filename="data/06_01_10.csv"
	my_data = genfromtxt(filename, delimiter=',')
	return my_data

def get_data(n_iter,batch_size,num_steps,percentTest=.2,random_state=42):
    #Gen the x data
    #Gen the y data
    
    #g = -.2394
    #min_val = 0.0
    #max_val = 100.0
    #num_steps

	num_entries = n_iter * batch_size
    x = np.zeros(shape=(num_entries,num_steps))

    #y = np.zeros(shape=(num_entries,1))
    for i in xrange(0,num_entries):
        #I will feed in two timesteps of the data
        y0 = random.uniform(0.0,.5)
        v = random.uniform(0.0,.5)# These should be random. 
        g = random.uniform(0.0,.5)
        t = random.uniform(0.0,.5)
        dt = random.uniform(0.0,.5)

        for j in xrange(0,num_steps):
        	#This is the time ticker
        	t = t+dt*j
	        x[i][j] = y0 + v * t - 0.5 * g * t * t
        #x[i] = [x1]
        #y[i] = [x12]

        #randNum1 = random.uniform(min_val, max_val)
        #randNum2 = randNum1#random.uniform(min_val, max_val)
        #x[i] = [randNum1,randNum2]
        #y[i] = [randNum1*randNum2]
    train_X = x

    #train_Y = y
    #print x
    #print y
    #X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    return X_train[0:-2],X_train[1:-1]#, y_train, X_val, y_val



def main():
	# --- Set data params ----------------
	#Create Data
	max_len_data = 1000000000

	n_iter = 100000
	batch_size = 
	data = get_data(n_iter,batch_size,num_steps)
	#data2 , standardDev,meanVal= file_data(filename="data/06_11_12.csv")

	n_input = len(data[0])

	n_output = n_input
	n_hidden = 100
	learning_rate = 0.01
	decay = 0.9
	numEpochs = 200
	#reuse = True

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

	cells = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias =1) for _ in range(num_layers)],state_is_tuple=True)

	#cells = core_rnn_cell_impl.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	if h == None:
		h = cells.zero_state(1,tf.float32)

	hidden_out, states = tf.nn.dynamic_rnn(cells, X, sequence_length=sequence_length, dtype=tf.float32,initial_state=h)


	# if h == None:
	# 	h = cell.zero_state(1,tf.int32)
	# hidden_out, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


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
		
		steps = []
		losses = []
		accs = []
		validation_losses = []
		curEpoch = 0

		
		training_state = None
		i = 0
		print ("Number train: " , len(data))
		train_file_name = "loss.csv"
		train_loss_file = open(train_file_name,'w')
		

		while curEpoch < numEpochs:
			i += 1
			#Batch sizes of 30
			if ((i+1) > (len(data)-1.0)/seq_len):
				i = 1
				curEpoch += 1
			myTrain_x = data[seq_len*i:seq_len*(i+1)].reshape((1,seq_len,n_input))
			myTrain_y = data[seq_len*i+1:seq_len*(i+1)+1].reshape((1,seq_len,n_input))

			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			empty,loss,training_state,output_data_2 = sess.run([optimizer,  cost, states,output_data], feed_dict = myfeed_dict)
			#print("TrainX: ", myTrain_x)
			
			print("Epoch: " + str(curEpoch) + " Iter " + str(i) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss))
			if (curEpoch % 5 == 0 and i ==1):
				outputVal = np.array(output_data_2*1.0+0.0)
				correctVal = myTrain_y
				#outputVal = np.array(output_data_2*standardDev+meanVal)
				#correctVal = myTrain_y*standardDev+meanVal
				print("Output: " , outputVal)
				print("My train: " , correctVal)
				#print("Output - myTrain: " , outputVal-correctVal)
				print("My loss: " , loss)
				
				train_loss_file.write(str(loss)+"\n")
				train_loss_file.flush()
				#do_validation(f2,data,curEpoch)
				file_name_file = saver.save(sess,os.path.join(savename,'my_model'))
				print("Model saved in: " , file_name_file)

		train_loss_file.close()
			

			
		print("Optimization Finished!")
		


if __name__=="__main__":
	
	main()
