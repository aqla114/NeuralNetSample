import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)

    return exp_x /sum_exp_x