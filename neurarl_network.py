import numpy as np
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error, numerical_gradient
from layers import Relu, Affine, SoftmaxWithLoss
from collections import OrderedDict

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, learning_rate=0.1):
        self.weight = {}
        self.weight['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.weight['b1'] = np.zeros(hidden_size)
        self.weight['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.weight['b2'] = np.zeros(output_size)

        self.learning_rate = 0.1

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.weight['W1'], self.weight['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.weight['W2'], self.weight['b2'])
        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        # W1, W2 = self.weight['W1'], self.weight['W2']
        # b1, b2 = self.weight['b1'], self.weight['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def train(self, x: np.ndarray, t: np.ndarray):
        grads = self.gradient_with_backpropagation(x, t)

        for key in self.weight.keys():
            self.weight[key] -= self.learning_rate * grads[key]

    def calc_loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)

        return self.last_layer.forward(y, t)
        # return cross_entropy_error(y, t)
    
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

    def gradient(self, x, t):
        W1, W2 = self.weight['W1'], self.weight['W2']
        b1, b2 = self.weight['b1'], self.weight['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def gradient_with_backpropagation(self, x, t):
        # forward
        self.calc_loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        # for key, value in grads.items():
        #     print(key + ' : ' + str(grads[key].shape))

        return grads
