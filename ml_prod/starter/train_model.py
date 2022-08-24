"""
Script to train machine learning model.
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from ml_prod.starter import common

# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25


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


def normalize_text(text: str) -> str:
    """ Normalize any text.

    Args:
        text:

    Returns:

    """
    return text.strip().lower().replace("-", "_")
    # return text.strip().lower()


def read_data(filename: str = common.path_dataset):
    """

    Returns:

    """
    df = pd.read_csv(filename)
    df = df.rename(columns=normalize_text)  # normalize column names
    normalized_text = df.select_dtypes(object).apply(lambda x: x.str.lower().str.strip().str.replace("-", "_"))
    df[df.select_dtypes(object).columns] = normalized_text
    return df


def process_data(dataframe: pd.DataFrame, *, categorical_features: List[str], label: str,
                 encoder_dict_: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """ Returns the input dataframe with categorical features and label encoded with the LabelEncoder defined in input.

    Args:
        dataframe: input dataset
        categorical_features: columns with the features
        label: target column
        encoder_dict_: dictionary with keys the name of the columns to encode (features and label) and values the
                      corresponding encoder

    Returns:
        encoded dataframe of features and label

    """
    output_df = dataframe.copy()
    for feature_ in categorical_features + [label]:
        encoder_ = encoder_dict_[feature_]
        # temp_encoder = LabelEncoder()
        # output_df[feature] = temp_encoder.fit_transform(dataframe[feature])
        output_df[feature_] = encoder_.transform(dataframe[feature_])

    return output_df


def get_encoder_dict(encoder_: LabelEncoder) -> Dict[str, int]:
    return {key: value for key, value in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def get_encoder_dict_inv(encoder_: LabelEncoder) -> Dict[int, str]:
    return {key: value for value, key in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def set_encoders(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                 save_encoders: bool = False) -> Dict[str, LabelEncoder]:
    """

    Args:
        dataframe:
        categorical_features:
        label:
        save_encoders: if True, save the encoders to a .pkl file

    Returns:

    """
    output_folder_ = common.path_model
    out_encoders = {}
    for feature_ in categorical_features + [label]:
        temp_encoder = LabelEncoder()
        temp_encoder.fit(dataframe[feature_])
        out_encoders.update({feature_: temp_encoder})

        if save_encoders:
            output_file = output_folder_ / f'enc_{feature_}.pkl'
            print(f"Saving encoder for {feature_} in {output_file}")
            with open(output_file, 'wb') as encoder_file:
                pickle.dump(temp_encoder, encoder_file)
    return out_encoders


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


_DO_TRAINING = True


if __name__ == '__main__':
    data = read_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]
    label_column = 'salary'
    cat_features = list(map(normalize_text, cat_features))

    encoders = set_encoders(dataframe=train, categorical_features=cat_features, label=label_column, save_encoders=True)
    train_encoded = process_data(train, categorical_features=cat_features, label=label_column,
                                 encoder_dict_=encoders)

    X_train = train_encoded[cat_features].to_numpy()
    y_train = train_encoded[label_column].to_numpy()

    if not _DO_TRAINING:
        exit()

    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model = Model()
    model.train(X_train, y_train)
    model.save(model_file)

    pred = model.predict(X_train)
    train_encoded['pred'] = pred
    train['pred'] = pred

    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(train_encoded, label_column, 'pred'))
    print(cm.confusion_matrix)
    print(cm.to_df())

    # slicing
    for feature in ['sex']:
        print(feature)
        for slice_, df_slice in train_encoded.groupby(feature):
            encoder_dict = get_encoder_dict_inv(encoders[feature])
            print(encoder_dict[slice_])
            cm_slice = common.BinaryConfusionMatrix(common.confusion_matrix_df(df_slice, label_column, 'pred'))
            print(cm.confusion_matrix)
            print(cm_slice.to_df())
            print('\n')
        print("--------------")
