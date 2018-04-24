import numpy as np
from neurarl_network import NeuralNetwork
from dataset.mnist import load_mnist
from deal_data import make_minibatch

def main():
    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=False, one_hot_label=True)

    network = NeuralNetwork(784, 100, 10)

    batch_size = 100

    for i in range(10000):
        # ミニバッチの取得
        batch_x, batch_t = make_minibatch(train_x, train_t, batch_size)

        network.train(batch_x, batch_t)

        loss = network.calc_loss(batch_x, batch_t)

        print('train {}: loss = {}'.format(i, loss))


if __name__ == '__main__':
    main()
