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

    def fit(self, inputs, labels):
        rgen = np.random.RandomState(self.seed)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(inputs.shape[1], 1))
        self.bias = rgen.normal(loc=0.0, scale=0.01)
        self.cost = []

        for epoch in range(self.epochs):
            outputs = self.activation(self.weighted_sum(inputs))
            errors = (labels - outputs)
            self.weights += self.learning_rate * inputs.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            self.cost.append((-labels.T.dot(np.log(outputs)) - (1 - labels.T).dot(np.log(1 - outputs)))[0][0])
        return self

    def predict(self, X):
        return np.where(self.activation(self.weighted_sum(X)) >= 0.5, 1, 0)