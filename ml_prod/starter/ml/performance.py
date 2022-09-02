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
        cm_dict = self.compute_performance_on_slice(feature)
        slice_perf = pd.concat(list(map(lambda x: x.to_df(), cm_dict.values())), ignore_index=True)
        slice_perf.index = cm_dict.keys()
        return slice_perf
