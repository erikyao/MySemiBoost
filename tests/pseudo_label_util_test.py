import os
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from sklearn.externals import joblib
from OSU18_data.prep import OSU18DataLoader
from MySemiBoost.msb.pseudo_label_util import pq_i
from MySemiBoost.msb.semi_booster import Ensemble


class PseudoLabelUtilCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        odr = OSU18DataLoader(data_dir=os.path.expanduser("~/Uexclave/semiboost-data"),
                              feat_fn="feature_matrix_osu18_2.tsv",
                              group_fn="group_id_osu18.tsv",
                              sim_fn="OSU18_S_mm_s1_np_float64_n39083.joblib")
        odr.load_features()
        odr.load_groups()
        odr.load_similarity_matrix()

        cls.X, cls.y, cls.g_X, cls.Z = odr.randomly_mask_groups(1200, 1337)

        cls.S = odr.sim_mat

        cls.XZ = pd.concat([cls.X, cls.Z])
        cls.H = Ensemble(t=0, index=cls.XZ.index)
        cls.C = cls.X.shape[0] / cls.Z.shape[0]

        cls.p_i_series = pd.read_csv("p_i_series.tsv", sep="\t", header=None, index_col=0).loc[:, 1]
        cls.q_i_series = pd.read_csv("q_i_series.tsv", sep="\t", header=None, index_col=0).loc[:, 1]

    # def save_pq_i(self):
    #     """
    #     Save p_i and q_i for other tests.
    #     Based on commit bfc9775247bf936f0e18f057844549fe4677c723
    #     """
    #     p_q_array = np.array([pq_i(i, self.y, self.Z, self.S, self.H, self.C) for i in self.XZ.index.values])
    #
    #     p_i_series = pd.Series(p_q_array[:, 0], index=self.XZ.index)
    #     q_i_series = pd.Series(p_q_array[:, 1], index=self.XZ.index)
    #
    #     p_i_series.to_csv("p_i_series.tsv", sep="\t", header=False, index=True)
    #     q_i_series.to_csv("q_i_series.tsv", sep="\t", header=False, index=True)

    def test_pq_i(self):
        p_q_array = np.array([pq_i(i, self.y, self.Z, self.S, self.H, self.C) for i in self.XZ.index.values])

        p_i_series = pd.Series(p_q_array[:, 0], index=self.XZ.index)
        q_i_series = pd.Series(p_q_array[:, 1], index=self.XZ.index)

        assert_series_equal(p_i_series, self.p_i_series, check_names=False)
        assert_series_equal(q_i_series, self.q_i_series, check_names=False)

if __name__ == '__main__':
    unittest.main()
