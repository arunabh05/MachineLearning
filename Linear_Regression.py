import random
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
	def __init__(self):
                self.X = np.empty([40,3])
                self.Y = np.empty([40])

	def run(self, X, Y):
		"""Compute pseudo-inverse of X and Y."""
		self.w = np.dot(np.linalg.pinv(X), Y)
		return self.w

	def classify(self, x):
		"""Classify 0 or 1"""
		return 1 if x > 0 else -1

	def generate_data(self):
            """Generate training data."""
            for i in range(40):
                self.X[i][0] = 1
                self.X[i][1] = random.uniform(-1, 1)
                self.X[i][2] = random.uniform(-1, 1)
                self.Y[i] = 1 if self.X[i][0] + self.X[i][1] + self.X[i][2] > 1 else -1
	    return self.X, self.Y
	

	def plotSet(self):
		for i in range(1, 40):	
			plt.plot(X[i,1],X[i,2], 'rx' if self.Y[i] == 1 else 'bx')


	def plot(self):
		plt.plot([-1,1],[(-self.w[0] - self.w[1] * -1) /self.w[2],(-self.w[0] - self.w[1] * 1 /self.w[2])],'--k')

	def predict(self, x):
		"""Predict output on x"""
		return self.classify(np.dot(np.transpose(self.w), x))

	def get_ein(self, X, Y):
		"""Compute Ein on X."""
		e_in = 0.0
		for i in range(X.shape[0]):
			if self.predict(X[i, :]) != Y[i]:
				e_in += 1
		return e_in / X.shape[0]

if __name__ == '__main__':
	lr = LinearRegression()
	X, Y = lr.generate_data()
	print lr.run(X, Y)
	lr.plotSet()
	plt.show()
	print lr.get_ein(X, Y)
	lr.plotSet()
	lr.plot()
	plt.show()
