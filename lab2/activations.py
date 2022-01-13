import numpy as np

def softmax(x, t=1):
    y = x - np.max(x, keepdims=True)
    dist = np.exp(y / t) / np.sum(np.exp(y / t), keepdims=True)
    return dist