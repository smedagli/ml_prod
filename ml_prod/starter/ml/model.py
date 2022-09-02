"""
TODO:
    - check if methods proposed from Udacity are completely covered
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    pass


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass


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
