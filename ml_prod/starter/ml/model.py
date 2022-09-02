"""
Contains the implementation of the model
TODO:
    - check if methods proposed from Udacity are completely covered
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


class Model:
    """
    TODO:
        - add docstring
    """
    def __init__(self):
        self.model = LogisticRegression()

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as output_file:
            pickle.dump(obj=self.model, file=output_file)

    def load(self, filename: str) -> None:
        with open(filename, 'rb') as input_file:
            self.model = pickle.load(input_file)

    def train(self, x: np.array, y: np.array) -> None:
        self.model.fit(X=x, y=y)

    def predict(self, x: np.array) -> np.array:
        return self.model.predict(x)
