import numpy as np


class LogisticRegressionGD(object):

    def __init__(self, learning_rate=0.05, epochs=100, seed=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def weighted_sum(self, X):
        return X.dot(self.weights) + self.bias

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        rgen = np.random.RandomState(self.seed)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1], 1))
        self.bias = rgen.normal(loc=0.0, scale=0.01)
        self.cost = []

        for _ in range(self.epochs):
            weighted_sum = self.weighted_sum(X)
            output = self.activation(weighted_sum)
            errors = (y - output)
            self.weights += self.learning_rate * X.T.dot(y - self.activation(self.weighted_sum(X)))
            self.bias += self.learning_rate * errors.sum()
            #cost = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))
            #self.cost.append(cost)
        return self

    def predict(self, X):
        return np.where(self.activation(self.weighted_sum(X)) >= 0.5, 1, 0)