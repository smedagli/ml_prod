import unittest
import numpy as np
import pandas as pd

from starter import common


class test_common(unittest.TestCase):
    def test_confusion_matrix_df(self):
        pred = np.array([0, 0, 1, 1])
        labels = np.array([1, 1, 1, 0])
        df = pd.DataFrame((pred, labels), index=['pred', 'label']).T
        cm = common.confusion_matrix_df(df, label='label', pred_column='pred')
        self.assertEqual(cm[0][0], 0)
        self.assertEqual(cm[0][1], 1)
        self.assertEqual(cm[1][0], 2)
        self.assertEqual(cm[1][1], 1)

    def test_binary_confusion_matrix(self):
        conf_matrix_np = np.array([[20, 3], [32, 12]])
        cm = common.BinaryConfusionMatrix(conf_matrix_np)
        self.assertAlmostEqual(cm.accuracy.round(2), 0.48)
        self.assertAlmostEqual(cm.f1_score.round(2), 0.41)
        self.assertEqual(cm.false_negatives, 32)
        self.assertEqual(cm.true_negative, 20)
        self.assertAlmostEqual(cm.recall.round(2), 0.27)
        self.assertAlmostEqual(cm.to_df()['recall'].iloc[0], cm.recall)


if __name__ == '__main__':
    unittest.main()
