import unittest
import pandas as pd
from pandas.util.testing import assert_series_equal
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from MySemiBoost.msb.semi_booster import SemiBooster
from MySemiBoost.msb.similarity_matrix import SimilarityMatrix


class BreastCancerTestCase(unittest.TestCase):
    def test_alpha(self):
        # data, label = load_breast_cancer(return_X_y=True)
        bc = load_breast_cancer()

        all_data = pd.DataFrame(bc['data'])
        all_data.columns = bc['feature_names']
        labels = pd.Series(bc['target']).replace(to_replace=0, value=-1)

        # Manually pick the first 300 entries as labeled data
        X = all_data.loc[0:299, ]  # contrary to usual python slices, both the start and the stop are included in .loc!
        y = labels.loc[0:299]
        Z = all_data.loc[300:, ]

        X = pd.DataFrame(MinMaxScaler().fit_transform(X=X), index=X.index, columns=X.columns)
        S = SimilarityMatrix.compute(all_data)

        lr_config = dict(penalty='l2', C=1.0, class_weight=None, random_state=1337,
                         solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)

        sb = SemiBooster(sigma=0.0000001,
                         T=20,
                         sample_percent=0.2,
                         base_classifier=LogisticRegression(**lr_config),
                         random_state=1337)
        sb.fit(X=X,
               y=y,
               Z=Z,
               S=S)

        train_scores = sb.train_scores()
        test_scores = sb.decision_function(all_data)

        # We calculated the scores on the same dataset, so they must be the same
        assert_series_equal(train_scores, test_scores)

        test_probs = sb.predict_proba(all_data)
        test_predits = sb.predict(all_data)

        import numpy as np
        index = np.where((test_probs[:, 1] > 0.5) != (test_predits == 1))
        print((test_probs[:, 1] > 0.5)[index])
        print((test_predits == 1)[index])

        self.assertTrue(all((test_probs[:, 1] > 0.5) == (test_predits == 1)))


if __name__ == '__main__':
    unittest.main()
