import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([])
        self.network['b1'] = np.array([])
        self.network['W2'] = np.array([])
        self.network['b2'] = np.array([])
        self.network['W3'] = np.array([])
        self.network['b3'] = np.array([])
    
    def forward(self, x):
        return x
