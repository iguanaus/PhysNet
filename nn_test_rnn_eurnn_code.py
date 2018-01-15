from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import *
import random

from numpy import zeros, newaxis


def copying_data(num_entries, n_sequence):
	#num_entries = n_iter * batch_size
    x = np.zeros(shape=(num_entries,n_sequence+1))
    for i in xrange(0,num_entries):
        #I will feed in two timesteps of the data
        y0 = random.uniform(0.0,.5)
        v = random.uniform(0.0,.5)# These should be random. 
        g = random.uniform(0.0,.5)
        t = random.uniform(0.0,.5)
        dt = random.uniform(0.0,.5)

        for j in xrange(0,n_sequence+1):
        	#This is the time ticker
        	t = t+dt*j
	        x[i][j] = y0 + v * t - 0.5 * g * t * t
        #x[i] = [x1]
        #y[i] = [x12]

        #randNum1 = random.uniform(min_val, max_val)
        #randNum2 = randNum1#random.uniform(min_val, max_val)
        #x[i] = [randNum1,randNum2]
        #y[i] = [randNum1*randNum2]
    X_train = x
    print (x[0])
    y= X_train[:,1:]
    x= X_train[:,0:-1]
    x = x[: , : , newaxis]
    y = y[: , : , newaxis]
    print (x[0])
    print(y[0])
    print(x.shape)

    return x,y#, y_train, X_val, y_val

def main(model, T, n_iter, n_batch, n_hidden, capacity, approx, init_state, save_result):
	print(model, T, n_iter, n_batch, n_hidden, capacity, approx, init_state, save_result)

	# --- Set data params ----------------
	n_sequence = 20
	num_entries = n_iter * n_batch

	n_input = 1
	n_output = 1
	n_classes = 1
	#n_sequence = 10
	#n_train = n_iter * n_batch
	#n_test = n_batch

	#n_input = 10
	#n_steps = T+20
	#n_classes = 9


  	# --- Create data --------------------
  	step1 = time.time()
  	
	train_x, train_y = copying_data(num_entries, n_sequence)
	test_x, test_y = copying_data(int(num_entries*.2), n_sequence)

	step2 = time.time()
	print("\n-- Prepare data: " + str(step2 - step1) + "\n")


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("float32", [None, n_sequence,1])
	y = tf.placeholder("float32", [None, n_sequence,1])
	
	input_data = x#tf.one_hot(x, n_input, dtype=tf.float32)
	print("X data: " , x)


	# initialization of hidden layer
	if init_state:
		init_val = np.sqrt(3./(2*n_hidden))
		h0 = tf.Variable(tf.random_uniform([1, n_hidden], minval=-init_val, maxval=init_val), name="h0")
		h = tf.tile(h0, [n_batch, 1])
	else:
		h = None


	# test normal rnn
	# input_data = tf.unpack(tf.transpose(input_data, [1,0,2]))


	# Input to hidden layer
	sequence_length = [n_sequence] * n_batch


	#cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

	layers = [tf.nn.rnn_cell.BasicRNNCell(size) for size in (10,10,5,10,10)]

	#cell = tf.nn.rnn_cell.BasicRNNCell(10)
	#cell2 = tf.nn.rnn_cell.BasicRNNCell(10)
	#4 Layers of 10. 
	cells = tf.contrib.rnn.MultiRNNCell(layers)
	print("Cells: " , cells)

	print("Input data : " , input_data)
	print("Seq len  : " , sequence_length)
	hidden_out, _ = tf.nn.dynamic_rnn(cells, input_data, sequence_length=sequence_length, dtype=tf.float32)



	# Hidden Layer to Output
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	
	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	print("Temp out copy: " , temp_out)
	print("Transposed temp: " , tf.transpose(temp_out,[1,0,2]))
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 
	print("Output data: " , output_data)
	# if data_type == "complex":
	# 	# print(hidden_out.get_shape())
	# 	# print(V_weights.get_shape())
	# 	# output_data = tf.nn.bias_add(tf.matmul(tf.real(hidden_out), V_weights), V_bias)
	# 	hidden_out = tf.unpack(hidden_out, axis=1)
	# 	output_data = tf.pack([tf.matmul(i, V_weights) + V_bias for i in hidden_out], axis=1)
	# elif data_type == "float":
	# 	hidden_out = tf.unpack(hidden_out, axis=1)
	# 	output_data = tf.pack([tf.matmul(i, V_weights) + V_bias for i in hidden_out], axis=1)

	# define evaluate process
	cost = tf.reduce_mean(tf.square(y-output_data))
	
	step3 = time.time()
	print("\n-- Create graph: " + str(step3 - step2) + "\n")

	# --- Initialization --------------------------------------------------
	global_step = tf.Variable(0, trainable=False)
	decay_cycle = 200
	#JOHN EDIT!!!! Uncomment the LR and change the deacy to 0.5
	learning_rate = .0001
	#learning_rate = tf.train.exponential_decay(0.001, global_step, decay_cycle, 0.5, staircase=False)
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9).minimize(cost, global_step=global_step)
	# optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
	init = tf.global_variables_initializer()
	
	step4 = time.time()
	print("\n-- initialize gradients: " + str(step4 - step3) + "\n")

	total_param = 0
	for i in tf.global_variables():

		print(i.name + ": " + str(i.get_shape()))
		temp = 1
		s = i.get_shape()
		for j in range(len(s)):
			temp = temp * int(s[j])
		total_param += temp
	print("Total parameter number: " + str(total_param))

	# --- baseline -----
	baseline = np.log(8) * 10/(T+20)
	print("Baseline is " + str(baseline))
	# --- Training Loop ---------------------------------------------------------------
	steps = []
	losses = []
	accs = []

	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
		print("Session Created")

		sess.run(init)

		step5 = time.time()
		print("\n-- Initialization: " + str(step5 - step4) + "\n")
		step = 0
		while step < n_iter:
			batch_x = train_x[step * n_batch : (step+1) * n_batch]
			batch_y = train_y[step * n_batch : (step+1) * n_batch]

			#print("Batch X: " , batch_x[0])
		
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			vals = sess.run(output_data, feed_dict={x:batch_x,y:batch_y})
			if (step % 1000 == 0):
				print("Vals: " , vals[0])
				print("Desired: " , batch_y[0])

			# output intermediate nodes
			# h1 = sess.run(hidden_out_list[0], feed_dict={x: batch_x, y: batch_y})
			# h2 = sess.run(hidden_out_list[-1], feed_dict={x: batch_x, y: batch_y})
			# print(h1[0][0])
			# print(h2[0][0])

			# if step % display_step == 0:
			acc = 0#sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print("Iter " + str(step) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))

			steps.append(step)
			losses.append(loss)
			accs.append(acc)
			step += 1

		print("Optimization Finished!")
		step6 = time.time()
		print("-- Training Time: " + str(step6 - step5))
		
		# --- test ----------------------
		#batch_x = test_x
		#batch_y = test_y
		
		#sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		#test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
		#test_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
		#print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
		#			", Accuracy= " + "{:.5f}".format(test_acc))
	

		# plt.plot(steps, losses, 'b', steps, len(steps) * [baseline], '--')	
		# plt.axis([0,n_iter,0,max(losses)])
		# plt.show()

	
			


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Copying Memory Problem")
	parser.add_argument("--model", default='LSTM', help='Model name: RNN, LSTM, URNN, EURNN')
	parser.add_argument("--T", type=int, default=10, help='Copying Problem delay')
	parser.add_argument("--n_iter", type=int, default=10000, help='training iteration number')
	parser.add_argument("--n_batch", type=int, default=128, help='batch size')
	parser.add_argument("--n_hidden", type=int, default=10, help='hidden layer size')
	parser.add_argument("--capacity", type=int, default=2, help='capacity, only for EURNN, default is 2')
	parser.add_argument("--approx", type=str, default="False", help='diagonal matrix, only for EURNN, default is False')

	parser.add_argument("--init_state", type=str, default="False", help='Bool type: Initial hidden state as variable')
	parser.add_argument("--save_result", type=str, default="False", help='Bool type: Save result as txt file')

	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'T': dict['T'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'approx': dict['approx'],

			  	'init_state': dict['init_state'],
			  	'save_result': dict['save_result']
				# 'learning_rate': np.float32(dict['learning_rate']),
			 #  	'decay_rate': np.float32(dict['decay_rate']),
			}
	print(kwargs)
	main(**kwargs)
