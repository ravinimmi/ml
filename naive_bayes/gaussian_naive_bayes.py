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
            s = np.zeros(X.shape[1])
            count = 0.0
            for j in range(len(X)):
                if y[j] == class_:
                    s += X[j]
                    count += 1
            self.mu[class_] = s / count

    # def _calculate_mu(self, X, y):
    #     for class_ in set(y):
    #         t = X[np.array(y) == class_]
    #         s = np.dot(np.ones((1, t.shape[0])), t)[0]
    #         self.mu[class_] = s / t.shape[0]

    def _calculate_sigma(self, X, y):
        for class_ in set(y):
            t = np.zeros(X.shape[1])
            count = 0.0
            for j in range(len(X)):
                if y[j] == class_:
                    t += (X[j] - self.mu[class_]) ** 2
                    count += 1
            self.sigma[class_] = np.sqrt(t / count)

    def _calculate_class_prob(self, x, class_):
        p = 1
        for i in range(len(x)):
            p *= self._calculate_feature_prob(x, i, class_)
        return p

    def _calculate_feature_prob(self, x, i, class_):
        epsilon = 10 ** -9
        mu = self.mu[class_][i]
        sigma = self.sigma[class_][i] + epsilon
        p = (x[i] - mu) / sigma
        return math.exp(p * p / -2) / math.sqrt(2 * math.pi * sigma * sigma)
