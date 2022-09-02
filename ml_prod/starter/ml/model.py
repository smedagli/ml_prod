"""
Contains the implementation of the model
TODO:
    - Optional: implement hyperparameter tuning.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


class Model:
    """
    TODO:
        - add docstring
    """
    def __init__(self, verbose: bool = False):
        self.model = LogisticRegression()
        self._verbose = verbose

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as output_file:
            pickle.dump(obj=self.model, file=output_file)
        if self._verbose:
            print(f"Saving model in {filename}")

    def load(self, filename: str) -> None:
        with open(filename, 'rb') as input_file:
            self.model = pickle.load(input_file)
        if self._verbose:
            print(f"Loading model: {filename}")

    def train(self, x: np.array, y: np.array) -> None:
        """ Training method """
        self.model.fit(X=x, y=y)

    def predict(self, x: np.array) -> np.array:
        """ Inference method """
        return self.model.predict(x)
