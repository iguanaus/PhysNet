import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.decomposition import PCA

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


    return weights , biases, embedding

output_folder = 'results/Project_12_00_acc/'
num_layers = 4
weights, biases, embedding = load_weights(output_folder,"",num_layers)
print("Weights: " , weights)
print("Biases: " , biases)


#This returns the 5 in the middle from the input
def forward(X):
    htemp = None
    #I will preset the architecture.
    print("X var: " , X)
    print(weights[0])
    print(biases[0])
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(X,weights[0])),biases[0])#Go into first 10 layer
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[1])),biases[1]) #Second 10
    htemp = tf.add(tf.nn.sigmoid(tf.matmul(htemp,weights[2])),biases[2])# Now 5
    return htemp


x_pre = tf.placeholder("int32", shape=[None, 3])

inputs = tf.nn.embedding_lookup(embedding, x_pre)

X1 = tf.reshape(inputs,[-1,300])

vals = forward(X1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #This will be a list of lists.
    data = np.zeros(shape=(2000,10))
    ij = 0
    while True:
        
        print(ij)
        if ij > 1999:
            break

        y0 = random.uniform(0.0,.5)
        v = random.uniform(0.0,.5)# These should be random. 
        g = random.uniform(0.0,.5)
        t = random.uniform(0.0,.5)
        dt = random.uniform(0.0,.5)

        x1 = y0 + v*t+-0.5*g*t*t
        x2 = y0 + v*(t+dt)+-0.5*g*(t+dt)*(t+dt)
        x3 = y0 + v*(t+2*dt)+-0.5*g*(t+2*dt)*(t+2*dt)
        x4 = y0 + v*(t+3*dt)+-0.5*g*(t+3*dt)*(t+3*dt)
        x = np.reshape(np.array([x1,x2,x3]),(1,-1))

        x_temp = [x1,x2,x3,x4]

        if (min(x_temp) > 0.0 and max(x_temp) < 1.0):
            maxNum = 100.0
            print(x)
            print(x.shape)

            for i in xrange(x.shape[0]):
                for j in xrange(x.shape[1]):
                    x[i][j] = int(x[i][j]*maxNum)#+500.
            print(x)

            mycost = sess.run(vals,feed_dict={x_pre:x})[0]

            #print("Vals: " , vals.eval())
            print("Mycost: " , mycost)
            data[int(ij)] = np.array([y0,v,g,t,dt,mycost[0],mycost[1],mycost[2],mycost[3],mycost[4]])
            ij += 1.0
    print(data)
    if False: #2D PCA

        #2 D PCA
        pca = PCA(n_components=2)

        projected = pca.fit_transform(data)
        print data.shape
        print projected
        plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('spectral', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        #plt.colorbar();
        plt.show()
    if True:
	    pca = PCA().fit(data)
	    plt.plot(np.cumsum(pca.explained_variance_ratio_))
	    plt.xlabel('number of components')
	    plt.ylabel('cumulative explained variance');
	    plt.show()



