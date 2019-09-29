import numpy as np


class AdalineGD(object):

    def __init__(self, learning_rate=0.01, epochs=50, seed=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def net_input(self, X):
        return X.dot(self.weights) + self.bias

    def activation(self, X):
        return X

    def fit(self, X, y):
        rgen = np.random.RandomState(self.seed)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1], 1))
        self.bias = rgen.normal(loc=0.0, scale=0.01)
        self.cost = []

        for epoch in range(self.epochs):
            output = self.activation(self.net_input(X))
            errors = (y - output)
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 200.0 # divide by big number to avoid overflow...
            self.cost.append(cost)
        return self

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0)