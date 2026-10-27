import numpy as np
from .implicit_als import ImplicitALS


class GradedALS(ImplicitALS):
    def preference(self, ratings):
        return np.maximum(ratings - 2, 0) / 3
