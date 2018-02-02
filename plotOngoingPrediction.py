#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np

#This returns 3 numbers and 6 lists. Namely in num_list_list, num_list_list, num_list_list

def read_file(filename):
	print('Reading: ' , filename)
	fo = open(filename,'r')
	line1 = fo.readline()
	line2 = fo.readline()
	line3 = fo.readline()
	vals = [float(x) for x in fo.readline()[:].split(',')]

	print(vals)
	a1 = [float(x) for x in fo.readline()[:].split(',')]
	p1 = [float(x) for x in fo.readline()[:].split(',')]
	print("Vals: " , vals)
	print("A1: " , a1)
	print("P1: " , p1)
	return vals,a1, p1	


legend = []

vals,a1,p1 = read_file("results/Project_11_size_20/test_out_file_single_22.txt")
plt.plot([i for i in xrange(0,len(vals))],vals,'o')
plt.plot([i+3 for i in xrange(0,len(a1))],a1,'o')
plt.plot([i+3 for i in xrange(0,len(p1))],p1,'o')
legend.append("Input")
legend.append("Actual")
legend.append("Predicted")

#plt.plot(range(400,802,2),a1)
#plt.plot(range(400,802,2),a1)
#plt.plot(range(400,802,2),p1)

plt.title('Projectile Motion Prediction')

plt.xlim(-1,len(vals)+len(p1)+1)

plt.ylabel("Y pos (m)")
plt.xlabel("Timestep")
plt.legend(legend, loc='top left')
plt.show()






