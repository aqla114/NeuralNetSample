import numpy as np
from functions import sigmoid, softmax

class NeuralNetwork():
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.random.rand(784, 100)
        self.network['b1'] = 1
        self.network['W2'] = np.random.rand(100, 10)
        self.network['b2'] = 1
    
    def predict(self, x):
        W1, W2 = self.network['W1'], self.network['W2']
        b1, b2 = self.network['b1'], self.network['b1']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
