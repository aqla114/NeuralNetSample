import numpy as np
from neurarl_network import NeuralNetwork
from dataset.mnist import load_mnist
from utils import make_minibatch

def main():
    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    network = NeuralNetwork(784, 50, 10)

    train_data_size = train_x.shape[0]
    batch_size = 100
    epoc_num = 10000
    train_num_per_epoc = max(int(train_data_size / batch_size), 1)

    for epoc in range(epoc_num):
        print('---------epoc {}------------'.format(epoc))

        # training
        for i in range(train_num_per_epoc): 
            # ミニバッチの取得
            batch_x, batch_t = make_minibatch(train_x, train_t, batch_size)

            network.train(batch_x, batch_t)

            loss = network.calc_loss(batch_x, batch_t)

            # print('train {}: loss = {}'.format(i, loss))
        
        # testing
        train_accuracy = network.calc_accuracy(train_x, train_t)
        test_accuracy = network.calc_accuracy(test_x, test_t)
        print('epoc {} : train_accuracy = {}, test_accuracy = {}'.format(epoc, train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
