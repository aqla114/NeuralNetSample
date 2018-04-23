import numpy as np
from neurarl_network import NeuralNetwork
from dataset.mnist import load_mnist

def main():
    network = NeuralNetwork()

    (train_x, train_t), (test_x, test_t) = load_mnist(flatten=True, normalize=False)

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(train_x), batch_size):
        batch_x = train_x[i:i+batch_size]
        batch_y = network.predict(batch_x)
        ans_list = np.argmax(batch_y, axis=1)
        # print('batch_x shape is {}, batch_y shape is {}'.format(batch_x.shape, batch_y,shape))
        accuracy_cnt += np.sum(ans_list == train_t[i:i + batch_size])

    print(accuracy_cnt)


if __name__ == '__main__':
    main()
