import numpy as np
from functions import sigmoid, softmax, cross_entropy_error, numerical_gradient

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.weight = {}
        self.weight['W1'] = np.random.rand(input_size, hidden_size)
        self.weight['b1'] = np.zeros(hidden_size)
        self.weight['W2'] = np.random.rand(hidden_size, output_size)
        self.weight['b2'] = np.zeros(output_size)

        self.learning_rate = 0.1
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        W1, W2 = self.weight['W1'], self.weight['W2']
        b1, b2 = self.weight['b1'], self.weight['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def train(self, x: np.ndarray, t: np.ndarray):
        grads = self.numerical_gradient(x, t)

        for key in self.weight.keys():
            self.weight[key] -= self.learning_rate * grads[key]
        

    def calc_loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def calc_accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        loss = lambda W: self.calc_loss(x, t)

        grads = {}

        for key in self.weight.keys():
            grads[key] = numerical_gradient(loss, self.weight[key])
        
        return grads
