"""
This module contains common tools, function and paths for the project
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from pathlib import Path

import ml_prod


_EPSILON_VAL = pow(10, -6)  # add this to resolve division by 0


class BinaryConfusionMatrix:
    def __init__(self, confusion_matrix_np: np.array):
        self.confusion_matrix = confusion_matrix_np

    @property
    def true_positives(self) -> int:
        return self.confusion_matrix[1, 1]

    @property
    def true_negative(self) -> int:
        return self.confusion_matrix[0, 0]

    @property
    def false_positives(self) -> int:
        return self.confusion_matrix[0, 1]

    @property
    def false_negatives(self) -> int:
        return self.confusion_matrix[1, 0]

    @property
    def accuracy(self) -> float:
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum().sum()

    @property
    def precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives + _EPSILON_VAL)

    @property
    def recall(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives + _EPSILON_VAL)

    @property
    def sensitivity(self) -> float:
        return self.recall

    @property
    def specificity(self) -> float:
        return self.true_negative / (self.true_negative + self.false_positives + _EPSILON_VAL)

    @property
    def f1_score(self) -> float:
        precision_and_recall = np.array([self.precision, self.recall])
        return 2 * np.prod(precision_and_recall) / np.sum(precision_and_recall + _EPSILON_VAL)

    @property
    def negative_predictive_value(self) -> float:
        return self.true_negative / (self.true_negative + self.false_negatives + _EPSILON_VAL)

    @property
    def positive_predictive_value(self) -> float:
        return self.precision

    def to_df(self) -> pd.DataFrame:
        data = [self.accuracy, self.recall, self.precision, self.sensitivity, self.specificity, self.f1_score,
                self.negative_predictive_value, self.positive_predictive_value,
                self.true_negative, self.true_positives, self.false_negatives, self.false_positives]
        indices = ['accuracy', 'recall', 'precision', 'sensitivity', 'specificity', 'f1_score',
                   'negative_predictive_value', 'positive_predictive_value',
                   'tn', 'tp', 'fn', 'fp']
        return pd.DataFrame(data, index=indices).T.fillna(0)


def confusion_matrix_df(dataframe: pd.DataFrame, label: str, pred_column: str = 'pred') -> np.array:
    """ Returns the confusion matrix for a given dataframe.
    It applies to binary classification problems.

    Args:
        dataframe: input dataframe
        label: target column
        pred_column: predictions' column

    Returns:

    """
    return metrics.confusion_matrix(y_true=dataframe[label].to_numpy(),
                                    y_pred=dataframe[pred_column].to_numpy(),
                                    labels=[False, True])


path_module = Path(ml_prod.__file__).parent.resolve()
path_dataset = path_module / 'data' / 'census.csv'
path_model = path_module / 'model'
