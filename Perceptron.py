import random
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
	
	def __init__(self,N=40):
		self.w = np.ones([3])
		self.w[0] = random.uniform(-1, 1)
		self.w[1] = random.uniform(-1, 1)
		self.w[2] = random.uniform(-1, 1)
		self.N = N
		self.X = np.empty([N,3])
		self.Y = np.empty([N])

	def generate_data(self):
	    """Generate training data."""
	    for i in range(self.N):
		self.X[i][0] = 1
		self.X[i][1] = random.uniform(-1, 1)
		self.X[i][2] = random.uniform(-1, 1)
		self.Y[i] = 1 if self.X[i][0] + self.X[i][1] + self.X[i][2] > 1 else -1

	    return self.X, self.Y
			
	def plotSet(self):
	    for i in range(self.N):
		plt.plot(self.X[i, 1],
		self.X[i, 2], 'ro' if self.Y[i] == 1 else 'bo')

	def classify(self, x):
		"""Classify 1 or 0"""
		return 1 if x > 0 else -1

	def response(self, x):
		"""Compute response"""
		print("product",self.w,x,np.dot(self.w,x))
		return self.classify(np.dot(self.w, x))

	def update_weights(self, x , error):
		"""Update weights."""
		self.w += error * x

	def train(self, X, Y):
		"""Run Perceptron Learning Algorithm"""
		done = False
		all_classified = False
		iteration = 0
		k = 0
		while not all_classified:
			all_classified = True
			for i in range(self.X.shape[0]):
				# print(Y[i] != self.response(X[i, :]))
				# print(Y[i])
				if Y[i] != self.response(X[i, :]):	
					error = Y[i] - self.response(X[i, :])
					self.update_weights(X[i, :], error)
					print(i, error)
					all_classified = False
	        	iteration += 1
		print("Done in %i iterations." % (iteration))
	
	def plot(self):
		plt.plot([-1,1],[(-self.w[0] - self.w[1] * -1) / self.w[2],(-self.w[0] - self.w[1] * 1) / self.w[2]], '--k')

if __name__ == '__main__':
	perceptron = Perceptron(100)
	X, Y = perceptron.generate_data()
	
	perceptron.plotSet()
	perceptron.plot()
	plt.show()

	perceptron.train(X, Y)
	perceptron.plotSet()
	perceptron.plot()
	plt.show()		
	
				


