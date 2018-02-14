import os
import unittest
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from OSU18_data.prep import prepare_data
from MySemiBoost.msb.pseudo_label_util import pq_i
from MySemiBoost.msb.semi_booster import Ensemble


class PseudoLabelUtilCase(unittest.TestCase):
    def test_pq_i(self):
        X, y, g_X, Z = prepare_data(1200, 1337)

        data_dir = os.path.expanduser("~/Uexclave/OSU18_MSB")
        joblib_fn = "OSU18_S_mm_s1_np_float64_n39083.joblib"
        S = joblib.load(os.path.join(data_dir, joblib_fn), "r")

        XZ = pd.concat([X, Z])
        H = Ensemble(t=0, index=XZ.index)
        C = X.shape[0] / Z.shape[0]

        p_q_array = np.array([pq_i(i, y, Z, S, H, C) for i in XZ.index.values])

        p_i_series = pd.Series(p_q_array[:, 0], index=XZ.index)
        q_i_series = pd.Series(p_q_array[:, 1], index=XZ.index)

        p_i_series.to_csv("p_i_series.tsv", sep="\t", header=False, index=True)
        q_i_series.to_csv("q_i_series.tsv", sep="\t", header=False, index=True)

if __name__ == '__main__':
    unittest.main()
