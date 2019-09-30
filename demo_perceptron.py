import numpy as np


class Perceptron(object):

    def __init__(self, learning_rate=0.01, epochs=100, seed=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def net_input(self, inputs):
        return inputs.dot(self.weights) + self.bias

    def fit(self, inputs, labels):
        rgen = np.random.RandomState(self.seed)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=(inputs.shape[1], 1))
        self.bias = rgen.normal(loc=0.0, scale=0.01)
        self.errors = []

        for epoch in range(self.epochs):
            errors = 0
            for input_sample, target_label in zip(inputs, labels):
                update = self.learning_rate * (target_label - self.predict(input_sample))
                self.weights += update * input_sample
                self.bias += update
                errors += int(update != 0)
            self.errors.append(errors)
        return self

    def predict(self, inputs):
        return np.where(self.net_input(inputs) >= 0.0, 1, 0)