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


if __name__ == '__main__':
    data_ = data.read_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data_, test_size=0.20)

    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                    "native-country"]
    label_column = 'salary'
    cat_features = list(map(data.normalize_text, cat_features))

    encoders = data.set_encoders(dataframe=train, categorical_features=cat_features, label=label_column, save_encoders=True)
    train_encoded = data.process_data(train, categorical_features=cat_features, label=label_column,
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
    train['label_encoded'] = train_encoded[label_column]

    cm = common.BinaryConfusionMatrix(common.confusion_matrix_df(train_encoded, label_column, 'pred'))
    print(cm.confusion_matrix)
    print(cm.to_df())

    evaluator = PerformanceEvaluator(dataframe=train, label_column='label_encoded', prediction_column='pred')
    for category in cat_features:
        print(category)
        print(evaluator.get_summary_slice(category))
        print('\n')
