import unittest
import copy
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from MySemiBoost.similarity_matrix import SimilarityMatrix


class SimilarityMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bc = load_breast_cancer()

        all_data = pd.DataFrame(bc['data'][0:10])
        all_data.columns = bc['feature_names']

        cls.X = pd.DataFrame(MinMaxScaler().fit_transform(all_data))

    def test_int_keys(self):
        S = SimilarityMatrix(self.X[0:4], sigma=1)
        D = S.dense_matrix

        for i in range(0, S.N):
            self.assertEqual(S[i, i], 1)

        for i in range(0, S.N):
            for j in range(0, S.N):
                self.assertEqual(S[i, j], S[j, i])

        """
        | i\j |   0  |   1  |   2  |   3  |
        |:---:|:----:|:----:|:----:|:----:|
        |  0  |   -  | D[0] | D[1] | D[2] |
        |  1  | D[0] |   -  | D[3] | D[4] |
        |  2  | D[1] | D[3] |   -  | D[5] |
        |  3  | D[2] | D[4] | D[5] |   -  |
        """
        self.assertEqual(S[1, 0], D[0])
        self.assertEqual(S[2, 0], D[1])
        self.assertEqual(S[3, 0], D[2])
        self.assertEqual(S[2, 1], D[3])
        self.assertEqual(S[3, 1], D[4])
        self.assertEqual(S[3, 2], D[5])

    def test_complex_keys(self):
        S = SimilarityMatrix(self.X, sigma=1)

        values_from_slice = S[0, :]
        self.assertEqual(len(values_from_slice), S.N)

        values_from_list = S[0, [i for i in range(0, S.N)]]
        self.assertEqual(len(values_from_list), S.N)

        values_from_array = S[0, np.arange(S.N)]
        self.assertEqual(len(values_from_array), S.N)

        values_from_bool = S[0, np.full(S.N, True)]
        self.assertEqual(len(values_from_bool), S.N)

        assert_array_equal(values_from_slice, values_from_list)
        assert_array_equal(values_from_slice, values_from_array)
        assert_array_equal(values_from_slice, values_from_bool)

    def test_access_diagonal(self):
        S = SimilarityMatrix(self.X, sigma=1)
        for i in range(0, S.N):
            self.assertEqual(S[i, i], 1)

    def test_change_sigma(self):
        sigma_1 = 2
        sigma_2 = 3

        S1 = SimilarityMatrix(self.X, sigma=sigma_1)

        S2 = copy.deepcopy(S1)
        S2.change_sigma(new_sigma=sigma_2)

        self.assertIsNot(S1, S2)

        """
        ln S1 / ln S2 == (sigma_2 ^ 2) / (sigma_1 ^ 2)
        """
        ratio = (sigma_2 ** 2) / (sigma_1 ** 2)
        for i in self.X.index.values:
            for j in self.X.index.values:
                if i != j:
                    # if i == j, np.log(S1[i, j]) / np.log(S2[i, j] == 0 / 0
                    self.assertAlmostEquals(np.log(S1[i, j]) / np.log(S2[i, j]), ratio)

        """
        A changed sim-matrix (S2 above) should be identical to a newly-created one (S3 below)
        """
        sigma_3 = sigma_2
        S3 = SimilarityMatrix(self.X, sigma=sigma_3)

        for i in self.X.index.values:
            for j in self.X.index.values:
                if i != j:
                    self.assertAlmostEquals(S2[i, j], S3[i, j])


if __name__ == '__main__':
    unittest.main()
