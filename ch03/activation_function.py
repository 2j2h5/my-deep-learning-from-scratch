import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y=x>0
    return y.astype(int)

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

""" x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)
y4 = softmax(x)
plt.plot(x, y1, x, y2, 'r-', x, y3, 'g-', x, y4, 'y-')
plt.ylim(-1, 5)
plt.show() """