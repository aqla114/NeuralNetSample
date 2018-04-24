import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    c = np.max(x)
    exp_x = np.exp(x - c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x

def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    batch対応版、cross_entropy_error。
    それぞれ、mini_batchを入力する。ただし、tはonehot。
    """
    # delta = 1e-7
    # return np.sum(t * np.log(y + delta))
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]

    return np.sum(t * np.log(y)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.shape[0]):
        tmp = x[index]

        # f(x + h)
        x[index] = tmp + h
        fx_upper = f(x)

        # f(x - h)
        x[index] = tmp - h
        fx_lower = f(x)

        grad[index] = (fx_upper - fx_lower) / (2*h)
        x[index] = tmp
    
    return grad
