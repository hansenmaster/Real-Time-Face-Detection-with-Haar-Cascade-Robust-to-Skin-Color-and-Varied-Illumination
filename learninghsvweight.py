# Back-Propagation Neural Networks
# dataset dapat di akses pada https://github.com/williamchand/Image-Digital-Processing
# program dapat langsung dicoba pada LiveDetection.py
# training dilakukan di google colaboratory https://colab.research.google.com/drive/1FxYoPk9ie-uhilXWxm9IM4Y0YYWQuuBd
# output dari training pada weights.txt
import math
import random
import pandas as pd
import random
random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
	def __init__(self, ni, nh, no):
		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# activations for nodes
		self.ai = [1.0]*self.ni
		self.ah = [1.0]*self.nh
		self.ao = [1.0]*self.no

		# create weights
		self.wi = makeMatrix(self.ni, self.nh)
		self.wo = makeMatrix(self.nh, self.no)
		# set them to random vaules
		for i in range(self.ni):
			for j in range(self.nh):
				self.wi[i][j] = rand(-0.2, 0.2)
		for j in range(self.nh):
			for k in range(self.no):
				self.wo[j][k] = rand(-2.0, 2.0)

		# last change in weights for momentum
		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	def update(self, inputs):
		if len(inputs) != self.ni-1:
			raise ValueError('wrong number of inputs')

		# input activations
		for i in range(self.ni-1):
			#self.ai[i] = sigmoid(inputs[i])
			self.ai[i] = inputs[i]

		# hidden activations
		for j in range(self.nh):
			sum = 0.0
			for i in range(self.ni):
				sum = sum + self.ai[i] * self.wi[i][j]
			self.ah[j] = sigmoid(sum)

		# output activations
		for k in range(self.no):
			sum = 0.0
			for j in range(self.nh):
				sum = sum + self.ah[j] * self.wo[j][k]
			self.ao[k] = sigmoid(sum)

		return self.ao[:]


	def backPropagate(self, targets, N, M):
		if len(targets) != self.no:
			raise ValueError('wrong number of target values')

		# calculate error terms for output
		output_deltas = [0.0] * self.no
		for k in range(self.no):
			error = targets[k]-self.ao[k]
			output_deltas[k] = dsigmoid(self.ao[k]) * error

		# calculate error terms for hidden
		hidden_deltas = [0.0] * self.nh
		for j in range(self.nh):
			error = 0.0
			for k in range(self.no):
				error = error + output_deltas[k]*self.wo[j][k]
			hidden_deltas[j] = dsigmoid(self.ah[j]) * error

		# update output weights
		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k]*self.ah[j]
				self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change
				#print N*change, M*self.co[j][k]

		# update input weights
		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j]*self.ai[i]
				self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
				self.ci[i][j] = change

		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5*(targets[k]-self.ao[k])**2
		return error


	def test(self, patterns):
		errno = 0
		for p in patterns:
			if (self.update(p[0])[0]<0.5):
				temp = 0
			else:
				temp = 1
			if (p[1][0] != temp):
				errno += 1
			print(p[0], '->', p[1], '->' ,self.update(p[0])[0])
			print (errno)
		self.weights()
			
	def weights(self):
		print('Input weights:\n')
		print(self.wi)
		print()
		print('Output weights:\n')
		print(self.wo)

	def train(self, patterns, iterations=500, N=0.01, M=0.001):
		# N: learning rate
		# M: momentum factor
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 100 == 0:
				print('error %-.5f' % error)


def demo():
	# Teach network XOR function
	# import dataset
	df = pd.read_csv('data.csv', delimiter=",")
	Hue = df['Hue'].tolist()
	Value = df['Value'].tolist()
	Output = df['Output'].tolist()
	pat=[]
	for temp1,temp2,temp3 in zip(Hue,Value,Output):
		pat += [[[temp1,temp2],[temp3]]]

	# create a network with two input, eleven hidden, and one output nodes
	n = NN(2, 11, 1)
	# train it with some patterns
	n.train(pat[:1800])
	# test it
	n.test(pat[:])
	
demo()