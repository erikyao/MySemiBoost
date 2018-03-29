import unittest
import pandas as pd
from pandas.util.testing import assert_series_equal
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from MySemiBoost.msb.semi_booster import SemiBooster
from MySemiBoost.msb.similarity_matrix import SimilarityMatrix


class BreastCancerTestCase(unittest.TestCase):
    def test_alpha(self):
        # data, label = load_breast_cancer(return_X_y=True)
        bc = load_breast_cancer()

        all_data = pd.DataFrame(bc['data'])
        all_data.columns = bc['feature_names']
        all_data = pd.DataFrame(MinMaxScaler().fit_transform(X=all_data),
                                index=all_data.index, columns=all_data.columns)

        labels = pd.Series(bc['target']).replace(to_replace=0, value=-1)

        # Manually pick the first 300 entries as labeled data
        X = all_data.loc[0:49, ]  # contrary to usual python slices, both the start and the stop are included in .loc!
        y = labels.loc[0:49]
        Z = all_data.loc[50:, ]

        S = SimilarityMatrix.compute(all_data)

        dt_config = dict(max_features="sqrt", random_state=1337, class_weight=None)
        dt = DecisionTreeClassifier(**dt_config)
        dm_config = dict(strategy='prior', random_state=1337)
        dm = DummyClassifier(**dm_config)

        sb = SemiBooster(sigma=1,
                         T=5,
                         sample_proportion=0.10,
                         C="XZ-ratio",
                         base_classifier=dt,
                         dummy_alpha="class-balance",
                         dummy_classifier=dm,
                         random_state=1337)
        sb.fit(X=X,
               y=y,
               Z=Z,
               S=S)

        train_scores = sb.train_scores()
        test_scores = sb.decision_function(all_data)

        # We calculated the decision_scores on the same dataset, so they must be the same
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
