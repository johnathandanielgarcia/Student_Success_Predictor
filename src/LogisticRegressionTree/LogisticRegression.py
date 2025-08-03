import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, max_it = 1000):
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.weight = None
        self.bias = 0

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))




