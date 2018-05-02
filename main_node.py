import numpy as np
from neurarl_network import NeuralNetwork
from dataset.mnist import load_mnist
from utils import make_minibatch
from layers import Affine, Relu
from nodes import SoftMaxWithCrossEntropyError, Mul, Add, Value, Relu

def main():
    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

    weight_init_std = 0.01

    W1 = Value(weight_init_std * np.random.randn(784, 50))
    b1 = Value(np.zeros(50))
    W2 = Value(weight_init_std * np.random.randn(50, 10))
    b2 = Value(np.zeros(10))

    train_data_size = train_x.shape[0]
    batch_size = 100
    epoc_num = 10000
    train_num_per_epoc = max(int(train_data_size / batch_size), 1)

    

    for epoc in range(epoc_num):
        print('---------epoc {}------------'.format(epoc))
        train_accuracy = 0

        # training
        for i in range(train_num_per_epoc): 
            # ミニバッチの取得
            batch_x, batch_t = make_minibatch(train_x, train_t, batch_size)

            network = SoftMaxWithCrossEntropyError(
                Add(
                    Mul(
                        Relu(
                            Add(
                                Mul(
                                    Value(batch_x),
                                    W1
                                    ),
                                b1
                                ),
                            ),
                        W2
                        ),
                    b2,
                ),
                Value(batch_t)
            )

            loss, acc = network.forward()
            train_accuracy += acc
            network.backward()

        network = SoftMaxWithCrossEntropyError(
            Add(
                Mul(
                    Relu(
                        Add(
                            Mul(
                                Value(test_x),
                                W1
                                ),
                            b1
                            ),
                        ),
                    W2
                    ),
                b2,
            ),
            Value(test_t)
        )

        # testing
        train_accuracy /= batch_size * train_num_per_epoc
        _, test_accuracy = network.forward()
        test_accuracy /= test_x.shape[0]
        print('epoc {} : train_accuracy = {}, test_accuracy = {}'.format(epoc, train_accuracy, test_accuracy))

    network.forward()

    network.backward()


if __name__ == '__main__':
    main()