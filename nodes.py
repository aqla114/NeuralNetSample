from layers import cross_entropy_error, softmax
import numpy as np
from dataset.mnist import load_mnist

class Add:
    def __init__(self, parent1, parent2):
        self.parent1 = parent1
        self.parent2 = parent2
        self.out = None
    
    def forward(self):
        out = self.parent1.forward() + self.parent2.forward()
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        self.parent1.backward(dx)
        self.parent2.backward(dy)

class Mul:
    def __init__(self, parent1, parent2):
        self.parent1 = parent1
        self.parent2 = parent2
        self.x = None
        self.y = None
    
    def forward(self):
        x = self.parent1.forward()
        y = self.parent2.forward()
        self.x = x
        self.y = y
        out = np.dot(x, y)
        
        return out
    
    def backward(self, dout):
        # print(dout.shape, self.x.shape, self.y.shape)
        dx = np.dot(dout, self.y.T)
        dy = np.dot(self.x.T, dout)
        self.parent1.backward(dx)
        self.parent2.backward(dy)

class Value:
    def __init__(self, value):
        self.value = value
        self.learning_rate = 0.1
    
    def forward(self):
        return self.value
    
    def backward(self, dout):
        self.update(dout)
    
    def update(self, dout):
        if dout.shape == self.value.shape:
            self.value -= self.learning_rate * dout
        else:
            self.value -= self.learning_rate * np.sum(dout, axis=0)

class Relu:
    def __init__(self, parent):
        self.parent = parent
    
    def forward(self):
        x = self.parent.forward()
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        self.parent.backward(dx)


class SoftMaxWithCrossEntropyError:
    def __init__(self, parent_x, parent_t):
        self.loss = None
        self.parent_x = parent_x
        self.parent_t = parent_t
        self.t = None
        self.y = None
        self.accuracy = 0
    
    def forward(self):
        x = self.parent_x.forward()
        self.t = self.parent_t.forward()
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        arg_t = np.argmax(self.t, axis=1)
        arg_y = np.argmax(self.y, axis=1)
        self.accuracy = np.sum(arg_t == arg_y)

        return self.loss, self.accuracy
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        self.parent_x.backward(dx)
