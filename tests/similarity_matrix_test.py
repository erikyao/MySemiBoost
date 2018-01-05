import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from MySemiBoost.similarity_matrix import similarity_matrix, similarity_matrix_percentiles, change_sigma


class SimilarityMatrixTestCase(unittest.TestCase):
    def test_change_sigma(self):
        bc = load_breast_cancer()

        all_data = pd.DataFrame(bc['data'][0:10])
        all_data.columns = bc['feature_names']

        sigma_1 = 2
        sigma_2 = 3

        preprocessor = preprocessing.MinMaxScaler()
        S1 = similarity_matrix(all_data, sigma=sigma_1, preprocessor=preprocessor)

        S2 = change_sigma(S1, cur_sigma=sigma_1, new_sigma=sigma_2)

        self.assertIsNot(S1, S2)

        """
        ln S1 / ln S2 == (sigma_2 ^ 2) / (sigma_1 ^ 2)
        """
        ratio = (sigma_2 ** 2) / (sigma_1 ** 2)
        for i in S1.index:
            for j in S1.columns:
                if i != j:
                    self.assertAlmostEquals(np.log(S1.loc[i, j]) / np.log(S2.loc[i, j]), ratio)

        """
        A changed sim-matrix (S2 above) should be identical to a newly-created one (S3 below)
        """
        sigma_3 = sigma_2
        S3 = similarity_matrix(all_data, sigma=sigma_3, preprocessor=preprocessor)

        for i in S2.index:
            for j in S2.columns:
                if i != j:
                    self.assertAlmostEquals(S2.loc[i, j], S3.loc[i, j])


if __name__ == '__main__':
    unittest.main()
