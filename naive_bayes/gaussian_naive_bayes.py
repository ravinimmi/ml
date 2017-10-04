import numpy as np
import math


class GaussianNB:
    def __init__(self):
        self.mu = {}
        self.sigma = {}
        self.prob = {}

    def fit(self, X, y):
        self._calculate_mu(X, y)
        self._calculate_sigma(X, y)

    def predict(self, X):
        prob = {}
        y = []
        for j in range(len(X)):
            for class_ in self.mu:
                prob[class_] = self._calculate_class_prob(X[j], class_)
            y.append(max(prob, key=prob.get))
        return y

    def score(self, X, y):
        pass

    def _calculate_mu(self, X, y):
        for class_ in set(y):
            t = X[np.array(y) == class_]
            self.mu[class_] = np.sum(t, axis=0) / t.shape[0]

    def _calculate_sigma(self, X, y):
        for class_ in set(y):
            t = X[np.array(y) == class_]
            row_count = t.shape[0]
            t = (t - self.mu[class_]) ** 2
            self.sigma[class_] = np.sqrt(np.sum(t, axis=0) / row_count)

    def _calculate_class_prob(self, x, class_):
        epsilon = 10 ** -9
        mu = self.mu[class_]
        sigma = self.sigma[class_] + epsilon

        return np.prod(
            np.exp(np.power((x - mu) / sigma, 2) / -2) /
            np.sqrt(2 * math.pi * sigma * sigma)
        )
