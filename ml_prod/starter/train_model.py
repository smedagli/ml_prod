"""
Script to train machine learning model.
"""

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

from ml_prod.starter import common
from ml_prod.starter.ml.model import Model
from ml_prod.starter.ml.performance import PerformanceEvaluator
from ml_prod.starter.ml import data

# TODO: remove this when complete
pd.options.display.width = 2500
pd.options.display.max_columns = 25

_DO_TRAINING = True
_VERBOSE = True


if __name__ == '__main__':
    data_ = data.read_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data_, test_size=0.20)  # split dataset

    cat_features = ["workclass", "education", "marital-status",
                    "occupation", "relationship", "race", "sex",
                    "native-country"]  # define categorical features
    label_column = 'salary'  # define label column
    cat_features = list(map(data.normalize_text, cat_features))

    # initialize model
    output_folder = Path(common.path_model)
    model_file = str((output_folder / 'model.pkl').resolve())
    model = Model(verbose=_VERBOSE)

    if _DO_TRAINING:  # train
        # TODO: check if dataframe has to be data_ or train (for set_encoders() )
        encoders = data.set_encoders(dataframe=data_, categorical_features=cat_features, label=label_column,
                                     save_encoders=True)  # initialize LabelEncoders for features and label

        output_prefix = 'train'  # will be the prefix of output performance files

        train_encoded = data.process_data(train, categorical_features=cat_features, label=label_column,
                                          encoder_dict_=encoders)  # encode the input dataset
        X_train = train_encoded[cat_features].to_numpy()
        y_train = train_encoded[label_column].to_numpy()
        # train
        model.train(X_train, y_train)
        model.save(model_file)
        pred = model.predict(X_train)

        dataset = train.copy()
        dataset_encoded = train_encoded.copy()
    else:  # test
        encoders = data.load_encoders(categorical_features=cat_features, label=label_column)

        output_prefix = 'test'  # will be the prefix of output performance files

        test_encoded = data.process_data(test, categorical_features=cat_features, label=label_column,
                                         encoder_dict_=encoders)  # encode the input dataset
        X_test = test_encoded[cat_features].to_numpy()
        y_test = test_encoded[label_column].to_numpy()
        # infer
        model.load(model_file)
        pred = model.predict(X_test)

        dataset = test.copy()
        dataset_encoded = test_encoded.copy()

    dataset['pred'] = pred
    dataset['label_encoded'] = dataset_encoded[label_column]
    dataset_encoded['pred'] = pred

    # performance evaluation
    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(dataset_encoded, label_column, 'pred'))
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
