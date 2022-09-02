"""
Tools and functions to pre-process data
TODO:
    - add implementation of "process_data" as proposed by Udacity
"""
import pickle
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from ml_prod.starter import common
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder


EncoderType = Union[LabelEncoder, LabelBinarizer, OneHotEncoder]


def process_data_udacity(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


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
                 encoder_dict_: Dict[str, EncoderType]) -> pd.DataFrame:
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


def get_encoder_dict(encoder_: EncoderType) -> Dict[str, int]:
    return {key: value for key, value in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def get_encoder_dict_inv(encoder_: EncoderType) -> Dict[int, str]:
    return {key: value for value, key in zip(encoder_.classes_, encoder_.transform(encoder_.classes_))}


def set_encoders(dataframe: pd.DataFrame, categorical_features: List[str], label: str, encoder_type: str = 'le',
                 save_encoders: bool = False) -> Dict[str, EncoderType]:
    """ Sets the encoders based on the encoder type specified

    Args:
        dataframe:
        categorical_features:
        label:
        encoder_type: 'le' for LabelEncoder, 'ohe' for OneHotEncoder
        save_encoders:

    Returns:

    """
    args = {'dataframe': dataframe, 'categorical_features': categorical_features,
            'label': label, 'save_encoders': save_encoders}
    if encoder_type:
        return set_encoders_le(**args)
    return set_encoders_ohe(**args)


def set_encoders_le(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                    save_encoders: bool = False) -> Dict[str, LabelEncoder]:
    """ Sets the encoders using LabelEncoder

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


def set_encoders_ohe(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                     save_encoders: bool = False) -> Dict[str, Union[LabelBinarizer, OneHotEncoder]]:
    """ Sets the encoders using OneHotEncoder

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
        if feature_ == label:
            temp_encoder = LabelBinarizer()
        else:
            temp_encoder = OneHotEncoder()
        temp_encoder.fit(dataframe[feature_])
        out_encoders.update({feature_: temp_encoder})

        if save_encoders:
            output_file = output_folder_ / f'enc_{feature_}.pkl'
            print(f"Saving encoder for {feature_} in {output_file}")
            with open(output_file, 'wb') as encoder_file:
                pickle.dump(temp_encoder, encoder_file)
    return out_encoders


def load_encoders(categorical_features: List[str], label: str) -> Dict[str, EncoderType]:
    """ Loads the encoders saved in training phase

    Args:
        categorical_features:
        label:

    Returns:

    """
    input_folder = common.path_model
    out_encoders = {}
    for feature_ in categorical_features + [label]:
        input_file = input_folder / f'enc_{feature_}.pkl'
        print(f"Loading encoder for {feature_} from {input_file}")

        with open(input_file, 'rb') as encoder_file:
            temp_encoder = pickle.load(encoder_file)
        out_encoders.update({feature_: temp_encoder})
    return out_encoders
