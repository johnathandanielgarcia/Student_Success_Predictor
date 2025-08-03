import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, max_it = 1000):
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.weight = None
        self.bias = None

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def fit(self, X, y):
        n_samples, n_feats, = X.shape
        self.weight = np.zeros(n_feats)
        self.bias = 0

        for i in range(self.max_it):
            lin_mod = np.dot(X, self.weight) + self.bias
            predicted = self.sigmoid(lin_mod)

            dw = (1 / n_samples) * np.dot(X.T, (predicted - y))
            db = (1 / n_samples) * np.sum((predicted - y))

            self.weight -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict_helper(self, X):
        lin_mod = np.dot(X, self.weight) + self.bias
        return self.sigmoid(lin_mod)

    def predict(self, X):
        prob = self.predict_helper(X)
        return np.where(prob >= 0.5, 1, 0)




