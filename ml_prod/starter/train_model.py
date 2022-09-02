"""
Script to train machine learning model.
TODO:
    - test case of OneHotEncoder
    - Optional enhancement, use K-fold cross validation instead of a train-test split.
"""
from sklearn.model_selection import train_test_split
from typing import List
from pathlib import Path
import pandas as pd

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model
from ml_prod.starter.ml.performance import PerformanceEvaluator
from ml_prod.starter.ml import data

# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25

_DO_TRAINING = False
_VERBOSE = True
_ENCODER_TYPE = 'le'


def train(model: Model, full_data: pd.DataFrame, train_data: pd.DataFrame, categorical_features: List[str], label: str,
          encoder_type: str = 'le') -> pd.DataFrame:
    common_args = {'categorical_features': categorical_features, 'label': label, 'encoder_type': encoder_type}
    encoders = data.set_encoders(dataframe=full_data, save_encoders=True,
                                 **common_args)  # initialize LabelEncoders for features and label

    x_train, y_train = data.process_data(train_data, encoder_dict_=encoders, **common_args)  # encode the input dataset
    # train
    model.train(x_train, y_train)
    model.save(model_file)
    pred = model.predict(x_train)

    out_dataset = train_data.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = y_train
    return out_dataset


def test(model: Model, test_data: pd.DataFrame, categorical_features: List[str], label: str,
         encoder_type: str = 'le') -> pd.DataFrame:
    common_args = {'categorical_features': categorical_features, 'label': label, 'encoder_type': encoder_type}
    encoder_dict = data.load_encoders(**common_args)

    x_test, y_test = data.process_data(test_data, encoder_dict_=encoder_dict, **common_args)
    # infer
    model.load(model_file)
    pred = model.predict(x_test)

    out_dataset = test_data.copy()
    out_dataset['pred'] = pred
    out_dataset['label_encoded'] = y_test
    return out_dataset


if __name__ == '__main__':
    data_ = data.read_data()

    train_df, test_df = train_test_split(data_, test_size=0.20)  # split dataset

    cat_features = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex",
                    "native-country"]  # define categorical features
    label_column = 'salary'  # define label column
    cat_features = list(map(data.normalize_text, cat_features))
    common_arguments = {'categorical_features': cat_features, 'label': label_column, 'encoder_type': _ENCODER_TYPE}

    # initialize model
    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model_ = Model(verbose=_VERBOSE_MODEL)

    if _DO_TRAINING:  # train
        output_prefix = 'train'  # will be the prefix of output performance files
        dataset = train(model=model_, full_data=data_, train_data=train_df, **common_arguments)
    else:  # test
        output_prefix = 'test'  # will be the prefix of output performance files
        dataset = test(model=model_, test_data=test_df, **common_arguments)

    # performance evaluation
    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(dataset, 'label_encoded', 'pred'))
    cm_output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{output_prefix}_confusion_matrix.csv')
    pd.DataFrame(cm.confusion_matrix).to_csv(cm_output_file)  # save confusion matrix

    if _VERBOSE:
        print(cm.confusion_matrix)
        print(cm.to_df())

    evaluator = PerformanceEvaluator(dataframe=dataset, label_column='label_encoded', prediction_column='pred')
    for category in cat_features:
        output_file = str(Path(common.path_module) / 'data' / 'performance' / f'{output_prefix}_{category}_slice.csv')
        if _VERBOSE:
            print(category)
            print(evaluator.get_summary_slice(category, output_file=output_file))
            print('\n')
