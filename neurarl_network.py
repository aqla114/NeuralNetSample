import numpy as np
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error, numerical_gradient
from layers import Relu, Tanh, Sigmoid, Affine, SoftmaxWithCrossEntropyError
from collections import OrderedDict

class NeuralNetwork():
    def __init__(self, weight_init_std=0.01, learning_rate=0.1):

        self.learning_rate = 0.1

        self.layers = OrderedDict()

        self.last_layer = SoftmaxWithCrossEntropyError()
    
    def append(self, layer_name, layer_type, input_size=None, output_size=None):
        if input_size == None or output_size == None:
            self.layers[layer_name] = layer_type()
        else:
            self.layers[layer_name] = layer_type(input_size, output_size)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def train(self, x: np.ndarray, t: np.ndarray):
        grads = self.gradient_with_backpropagation(x, t)

        for key in self.layers.keys():
            if (isinstance(self.layers[key], Affine)):
                self.layers[key].W -= grads[key]['W']
                self.layers[key].b -= grads[key]['b']

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

        for key in self.layers.keys():
            if (isinstance(self.layers[key], Affine)):
                grads[key] = {'W': self.layers[key].dW, 'b': self.layers[key].db}

        return grads

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

