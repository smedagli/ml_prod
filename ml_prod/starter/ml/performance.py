"""
This module contains tools and functions to monitor performance of a model
"""
from typing import Dict, Any, Union
import pandas as pd
from pathlib import Path

from ml_prod.starter import common
from sklearn.metrics import fbeta_score, precision_score, recall_score


class PerformanceEvaluator:
    def __init__(self, dataframe: pd.DataFrame, label_column: str, prediction_column: str = 'pred'):
        """

        Args:
            dataframe: the input dataset. It must include the label and prediction columns
            label_column: column of the binary encoded labels
            prediction_column: column of the binary encoded predictions
        """
        self.df = dataframe
        self.label_column = label_column
        self.prediction_column = prediction_column

    def compute_performance_on_slice(self, feature: str) -> Dict[Any, common.BinaryConfusionMatrix]:
        """ Slices the data based on the input feature and compute the common.BinaryConfusionMatrix for each slice

        Args:
            feature: the feature (column name) used to generate the data slices

        Returns:
            a confusion matrix for each category of the selected feature.
            The output is provided as dictionary with
            {key: value} ---> {category for the slice: confusion matrix on the category}

        """
        label_column = self.label_column
        prediction_column = self.prediction_column
        return {x[0]: common.BinaryConfusionMatrix(common.confusion_matrix_df(x[1], label_column, prediction_column))
                for x in self.df.groupby(feature)}

    def get_summary_slice(self, feature: str, output_file: Union[str, None]) -> pd.DataFrame:
        """ Merges together the performance on each category for the input feature.
        Each category has a different number of samples.
        To get the total number of sample for each category is sufficient to sum up the number of true/false
        positives/negatives (in the output dataframe ['tp', 'fp', 'tn', 'fn']

        Args:
            feature: the feature (column name) used to generate the data slices
            output_file: if not None, writes the summary to the specified file

        Returns:
            the summary of the performance for each category of the given slice (as pd.DataFrame)

        """
        cm_dict = self.compute_performance_on_slice(feature)
        slice_perf = pd.concat(list(map(lambda x: x.to_df(), cm_dict.values())), ignore_index=True)
        slice_perf.index = cm_dict.keys()

        if output_file:
            if not Path(output_file).parent.is_dir():
                Path(output_file).mkdir(parents=True)
            slice_perf.to_csv(output_file)
        return slice_perf


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
