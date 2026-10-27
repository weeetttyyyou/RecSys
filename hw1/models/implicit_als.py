import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Model, to_matrix


class ImplicitALS(Model):
    def __init__(self, dimensions=32, regularization=.5, alpha=10, epochs=10, seed=17):
        self.dimensions = dimensions
        self.regularization = regularization
        self.alpha, self.epochs, self.seed = alpha, epochs, seed

    def preference(self, ratings):
        return (ratings >= 4).astype(float)

    def fit(self, reviews, shape):
        observed = to_matrix(reviews, shape)
        preference = to_matrix(reviews, shape, self.preference(reviews.rating.to_numpy()))
        generator = np.random.default_rng(self.seed)
        self.book_factors = generator.normal(scale=.05, size=(shape[1], self.dimensions))
        for epoch in range(self.epochs):
            self.reader_factors = self.update(observed, preference, self.book_factors)
            self.book_factors = self.update(observed.T.tocsr(), preference.T.tocsr(), self.reader_factors)
        return self

    def update(self, observed, preference, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.dimensions)
        result = np.zeros((observed.shape[0], self.dimensions))
        for index in range(observed.shape[0]):
            begin, end = observed.indptr[index:index+2]
            selected = fixed[observed.indices[begin:end]]
            lhs = base + self.alpha * selected.T @ selected
            rhs = selected.T @ ((1 + self.alpha) * preference.data[begin:end])
            result[index] = cho_solve(cho_factor(lhs, check_finite=False), rhs, check_finite=False)
        return result

    def score(self, readers):
        return self.reader_factors[readers] @ self.book_factors.T
