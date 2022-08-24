import unittest

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ml_prod.starter import train_model


class test_train_model(unittest.TestCase):
    def test_column_names(self):
        self.assertTrue(train_model.read_data().columns.tolist()[: 3] == ['age', 'workclass', 'fnlgt'])

    def test_normalize_text(self):
        self.assertTrue(train_model.normalize_text('  sTa T-us') == 'sta t_us')
        self.assertTrue(train_model.normalize_text('AAAAAAA') == 'AAAAAAA'.lower())

    def test_get_encoder_dict(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c', 'd'])
        encoder_dict = train_model.get_encoder_dict(encoder)
        self.assertTrue(list(encoder_dict.keys()) == ['a', 'b', 'c', 'd'])
        self.assertTrue(list(encoder_dict.values()) == [0, 1, 2, 3])

    def test_get_encoder_dict_inv(self):
        encoder = LabelEncoder()
        encoder.fit(['a', 'b', 'c', 'd'])
        encoder_dict = train_model.get_encoder_dict_inv(encoder)
        self.assertTrue(list(encoder_dict.values()) == ['a', 'b', 'c', 'd'])
        self.assertTrue(list(encoder_dict.keys()) == [0, 1, 2, 3])

    def test_read_data(self):
        data = train_model.read_data()
        self.assertTrue(data.columns.tolist()[: 4] == ['age', 'workclass', 'fnlgt', 'education'])
        self.assertEqual(data.iloc[0]['age'], 39)

    def test_process_data(self):
        values = ['a', 'b', 'c', 'd', 'e']
        labels = ['yes', 'no', 'no', 'yes', 'yep']
        data = pd.DataFrame({'values': values, 'labels': labels})

        encoder_values = LabelEncoder()
        encoder_values.fit(data['values'])
        encoder_labels = LabelEncoder()
        encoder_labels.fit(data['labels'])
        encoder_dict = {'values': encoder_values, 'labels': encoder_labels}

        df_encoded = train_model.process_data(data, categorical_features=['values'], label='labels',
                                              encoder_dict_=encoder_dict)
        self.assertTrue(df_encoded['values'].tolist() == [0, 1, 2, 3, 4])
        self.assertTrue(df_encoded['labels'].tolist() == [2, 0, 0, 2, 1])


if __name__ == '__main__':
    unittest.main()
