import numpy as np
import math


class GaussianNB:
    def __init__(self):
        self.mu = {}
        self.sigma = {}
        self.prior_prob = {}

    def fit(self, X, y):
        self._calculate_mu(X, y)
        self._calculate_sigma(X, y)
        self._calculate_prior_prob(X, y)

    def predict(self, X):
        return [self._get_predicted_class(row) for row in X]

    def _calculate_mu(self, X, y):
        for class_ in set(y):
            t = X[np.array(y) == class_]
            self.mu[class_] = np.sum(t, axis=0) / t.shape[0]

    def _calculate_sigma(self, X, y):
        for class_ in set(y):
            t = X[np.array(y) == class_]
            self.sigma[class_] = np.sqrt(
                np.sum((t - self.mu[class_]) ** 2, axis=0) / t.shape[0]
            )

    def _calculate_prior_prob(self, X, y):
        for class_ in set(y):
            self.prior_prob[class_] = (
                X[np.array(y) == class_].shape[0] / X.shape[0]
            )

    def _get_predicted_class(self, x):
        prob = {}
        for class_ in self.mu:
            prob[class_] = (
                self.prior_prob[class_] * self._calculate_class_prob(x, class_)
            )
        return max(prob, key=prob.get)

    def _calculate_class_prob(self, x, class_):
        epsilon = 10 ** -9
        mu = self.mu[class_]
        sigma = self.sigma[class_] + epsilon

        return np.prod(
            np.exp(np.power((x - mu) / sigma, 2) / -2) /
            np.sqrt(2 * math.pi * sigma * sigma)
        )
