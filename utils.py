import numpy as np
from neurarl_network import NeuralNetwork
from dataset.mnist import load_mnist
from functions import numerical_gradient

def make_minibatch(datas_x, datas_t, batch_size):
    """
    datas_xにはdata配列を、datas_tにはその正解ラベル（ワンホット）配列を入れる。
    datas_x[mask], datas_t[mask] (それぞれbatch_size組のデータ)を返す。
    """
    train_size = datas_x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)
    
    return datas_x[batch_mask], datas_t[batch_mask]

def check_grad():
    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    network = NeuralNetwork(784, 50, 10)

    batch_x = train_x[:3]
    batch_t = train_t[:3]

    grad_numerical = network.numerical_gradient(batch_x, batch_t)
    grad_backpropagation =network.grad_with_backpropagation(batch_x ,batch_t)

    print(grad_numerical['W1'].shape, grad_backpropagation['W1'].shape)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backpropagation[key] - grad_numerical[key]))
        print(key + ":" + str(diff))


def check_numerical_grad():
    def func(x):
        return x**2

    ans = numerical_gradient(func, np.array([[5], [5]]))
    print(ans)
