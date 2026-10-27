import numpy as np
from scipy.sparse import csr_matrix
from .base import Model, to_matrix


class UserCF(Model):
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, data, shape):
        self.positive = to_matrix(data, shape, (data.rating >= 4).astype(float))
        lengths = np.sqrt(np.asarray(self.positive.sum(axis=1)).ravel())
        indices, values = [], []
        for first in range(0, shape[0], 128):
            last = min(first + 128, shape[0])
            overlap = (self.positive[first:last] @ self.positive.T).toarray()
            scores = overlap / np.maximum(lengths[first:last, None] * lengths[None, :], 1e-12)
            scores[np.arange(last-first), np.arange(first,last)] = 0
            top = np.argsort(-scores, axis=1, kind='stable')[:, :self.neighbors]
            weights = np.take_along_axis(scores, top, axis=1)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            indices.append(top)
            values.append(weights)
        columns = np.vstack(indices)
        rows = np.broadcast_to(np.arange(shape[0])[:, None], columns.shape)
        self.neighborhood = csr_matrix((np.vstack(values).ravel(), (rows.ravel(), columns.ravel())), shape=(shape[0], shape[0]))
        return self

    def score(self, users):
        return (self.neighborhood[users] @ self.positive).toarray()
