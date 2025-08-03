import numpy as np

'''
LogisticRegression object: 
self.learning_rate: the learning rate for the gradient descent updates
self.max_it = number of iterations 
self.weight = vector of weights; feature coefficients
self.bias = for linear combination
'''
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, max_it = 1000):
        self.learning_rate = learning_rate
        self.max_it = max_it
        self.weight = None
        self.bias = None
    '''
    Purpose: Sigmoid function that converts raw scores into probabilities
    Inputs: A numpy.ndarray or float
    Returns: numpy.ndarray containing the sigmoid-transformed probabilities
    '''
    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    '''
    Purpose: Train logistic regression model utilizing gradient descent 
    Inputs: 
        X: numpy.ndarray that is shape - training features
        y: numpy.ndarray that is samples - training labels; binary so 0 or 1
    How It Works: 
        1. Initialize bias (to 0) and weights
        2. Use sigmoid for predicted probabilities in each iteration 
        3. Calculate gradients (dw, db) 
        4. Use learning rate and gradients to update weights and bias
    '''
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
    '''
    Purpose: helper method to compute sigmoid probabilities on new data
    Input: numpy.ndarray 
    Returns: numpy.ndarray of predicted probabilities
    '''
    def predict_helper(self, X):
        lin_mod = np.dot(X, self.weight) + self.bias
        return self.sigmoid(lin_mod)
    '''
    Purpose: Predict binary class labels for given feature data
    Inputs: X, which is a numpy.ndarray 
    Returns: numpy.ndarray containing the predicted labels that are 0 or 1
    '''
    def predict(self, X):
        prob = self.predict_helper(X)
        return np.where(prob >= 0.5, 1, 0)




