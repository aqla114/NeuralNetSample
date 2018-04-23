import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)

    return exp_x /sum_exp_x

def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    batch対応版、cross_entropy_error。
    それぞれ、mini_batchを入力する。
    """
    # delta = 1e-7
    # return np.sum(t * np.log(y + delta))
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
    return np.sum(t * np.log(y)) / batch_size
