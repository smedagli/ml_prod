"""
Tools and functions to pre-process data
TODO:
    - add implementation of "process_data" as proposed by Udacity
"""
import pickle
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
from ml_prod.starter import common
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder


EncoderType = Union[LabelEncoder, LabelBinarizer, OneHotEncoder]


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


def process_data(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                 encoder_dict_: Dict[str, EncoderType], encoder_type: str = 'le') -> Tuple[np.array, np.array]:
    """ Returns the input dataframe with categorical features and label encoded with the LabelEncoder defined in input.

    Args:
        dataframe: input dataset
        categorical_features: columns with the features
        label: target column
        encoder_dict_: dictionary with keys the name of the columns to encode (features and label) and values the
                      corresponding encoder
        encoder_type:

    Returns:
        encoded dataframe of features and label

    """
    args = {'dataframe': dataframe, 'categorical_features': categorical_features, 'label': label,
            'encoder_dict': encoder_dict_}
    if encoder_type == 'le':
        return _process_data_le(**args)
    return _process_data_ohe(**args)


def _process_data_ohe(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                      encoder_dict: Dict[str, EncoderType]):
    """ Data pre-processing for OneHotEncoder

    Returns:

    """
    # scaler = StandardScaler()
    enc_features = encoder_dict['categorical'].transform(dataframe[categorical_features])
    enc_labels = encoder_dict['label'].transform(dataframe[label])

    numeric_features = dataframe.drop(categorical_features + [label], axis=1).values
    return np.concatenate([enc_features, numeric_features], axis=1), enc_labels


def _process_data_le(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
                     encoder_dict: Dict[str, EncoderType]) -> Tuple[np.array, np.array]:
    """ Data pre-processing for LabelEncoder

    Args:
        dataframe:
        categorical_features:
        label:
        encoder_dict: dictionary with keys the name of the columns to encode (features and label) and values the
                      corresponding encoder

    Returns:

    """
    output_df = dataframe.copy()
    for feature_ in categorical_features + [label]:
        encoder_ = encoder_dict[feature_]
        output_df[feature_] = encoder_.transform(dataframe[feature_])
    x = output_df[categorical_features].values
    y = output_df[label].values

    numeric_features = dataframe.drop(categorical_features + [label], axis=1).values
    return np.concatenate([x, numeric_features], axis=1), y


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
    if encoder_type == 'le':
        return _set_encoders_le(**args)
    return _set_encoders_ohe(**args)


def _set_encoders_le(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
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


def _set_encoders_ohe(dataframe: pd.DataFrame, categorical_features: List[str], label: str,
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

    cat_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cat_encoder.fit(dataframe[categorical_features])

    label_encoder = LabelBinarizer()
    label_encoder.fit(dataframe[label].to_numpy().reshape(-1, 1))

    out_encoders = {'categorical': cat_encoder, 'label': label_encoder}
    if save_encoders:
        category_output_file = output_folder_ / f'ohe_cat_encoders.pkl'
        label_output_file = output_folder_ / f'ohe_label_encoders.pkl'
        print(f"Saving OneHotEncoders for categories in in {category_output_file}")

        with open(category_output_file, 'wb') as encoder_file:
            pickle.dump(cat_encoder, encoder_file)

        print(f"Saving OneHotEncoders for label in in {label_output_file}")
        with open(label_output_file, 'wb') as encoder_file:
            pickle.dump(label_encoder, encoder_file)
    return out_encoders


def load_encoders(categorical_features: List[str], label: str, encoder_type: str = 'le') -> Dict[str, EncoderType]:
    if encoder_type == 'le':
        return load_encoders_le(categorical_features, label)
    return load_encoders_ohe()


def load_encoders_le(categorical_features: List[str], label: str) -> Dict[str, LabelEncoder]:
    """ Loads the encoders saved in training phase (for LabelEncoder) """
    input_folder = common.path_model
    out_encoders = {}
    for feature_ in categorical_features + [label]:
        input_file = input_folder / f'enc_{feature_}.pkl'
        print(f"Loading encoder for {feature_} from {input_file}")

        with open(input_file, 'rb') as encoder_file:
            temp_encoder = pickle.load(encoder_file)
        out_encoders.update({feature_: temp_encoder})
    return out_encoders


def load_encoders_ohe() -> Dict[str, EncoderType]:
    """ Loads the encoders saved in traning phase (for OneHotEncoder) """
    input_folder = common.path_model
    category_input_file = input_folder / 'ohe_cat_encoders.pkl'
    label_input_file = input_folder / 'ohe_label_encoders.pkl'

    with open(category_input_file, 'rb') as encoder_file:
        categorical_encoder = pickle.load(encoder_file)
    with open(label_input_file, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return {'categorical': categorical_encoder, 'label': label_encoder}
