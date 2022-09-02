"""
This module contains tools and functions to monitor performance of a model
"""
from typing import Dict, Any
import pandas as pd

from ml_prod.starter import common


class PerformanceEvaluator:
    def __init__(self, dataframe: pd.DataFrame, label_column: str, prediction_column: str = 'pred'):
        self.df = dataframe
        # self.feature = feature
        self.label_column = label_column
        self.prediction_column = prediction_column

    def compute_performance_on_slice(self, feature: str) -> Dict[Any, common.BinaryConfusionMatrix]:
        """ Slices the data based on the input feature and compute the common.BinaryConfusionMatrix for each slice

        Args:
            feature:

        Returns:

        """
        label_column = self.label_column
        prediction_column = self.prediction_column
        return {x[0]: common.BinaryConfusionMatrix(common.confusion_matrix_df(x[1], label_column, prediction_column))
                for x in self.df.groupby(feature)}

    def get_summary_slice(self, feature: str) -> pd.DataFrame:
        """

        Args:
            feature:

        Returns:

        """
        cm_dict = compute_performance_on_slice(self.df, feature, self.label_column, self.prediction_column)
        slice_perf = pd.concat(list(map(lambda x: x.to_df(), cm_dict.values())), ignore_index=True)
        slice_perf.index = cm_dict.keys()
        return slice_perf


def compute_performance_on_slice(dataframe: pd.DataFrame, feature: str, label_column: str,
                                 prediction_column: str = 'pred') -> Dict[Any, common.BinaryConfusionMatrix]:
    """ Slices the data based on the input feature and compute the common.BinaryConfusionMatrix for each slice/.

    Args:
        dataframe:
        feature:
        label_column:
        prediction_column:

    Returns:

    """
    return {x[0]: common.BinaryConfusionMatrix(common.confusion_matrix_df(x[1], label_column, prediction_column)) for x
            in dataframe.groupby(feature)}


def summary_slice(dataframe: pd.DataFrame, feature: str, label_column: str,
                  prediction_column: str = 'pred') -> pd.DataFrame:
    """

    Args:
        dataframe:
        feature:
        label_column:
        prediction_column:

    Returns:

    """
    cm_dict = compute_performance_on_slice(dataframe, feature, label_column, prediction_column)
    slice_perf = pd.concat(list(map(lambda x: x.to_df(), cm_dict.values())), ignore_index=True)
    slice_perf.index = cm_dict.keys()
    return slice_perf
