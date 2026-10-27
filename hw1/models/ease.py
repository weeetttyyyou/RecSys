import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Model, to_matrix


class EASE(Model):
    def __init__(self, regularization=250):
        self.regularization = regularization

    def fit(self, reviews, shape):
        self.x = to_matrix(reviews, shape, (reviews.rating >= 4).astype(float))
        covariance = (self.x.T @ self.x).toarray()
        covariance.flat[::shape[1] + 1] += self.regularization
        inverse = cho_solve(cho_factor(covariance), np.eye(shape[1]))
        self.coefficients = -inverse / inverse.diagonal()[None, :]
        np.fill_diagonal(self.coefficients, 0)
        return self

    def score(self, readers):
        return self.x[readers] @ self.coefficients
