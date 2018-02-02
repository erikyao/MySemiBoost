import unittest
import pandas as pd
from pandas.util.testing import assert_series_equal
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from MySemiBoost.semi_booster import SemiBooster
from MySemiBoost.similarity_matrix import SimilarityMatrix


class BreastCancerTestCase(unittest.TestCase):
    def test_alpha(self):
        # data, label = load_breast_cancer(return_X_y=True)
        bc = load_breast_cancer()

        all_data = pd.DataFrame(bc['data'])
        all_data.columns = bc['feature_names']
        labels = pd.Series(bc['target']).replace(to_replace=0, value=-1)

        # Manually pick the first 300 entries as labeled data
        labeled_data = all_data.loc[0:299, ]
        labels = labels[0:300]
        unlabeled_data = all_data.loc[300:, ]

        X = MinMaxScaler().fit_transform(X=all_data)
        S = SimilarityMatrix.compute(X)

        lr_config = dict(penalty='l2', C=1.0, class_weight=None, random_state=1337,
                         solver='liblinear', max_iter=100, verbose=0, warm_start=False, n_jobs=1)

        sb = SemiBooster(unlabeled_feat=unlabeled_data,
                         sigma=6.72488541e-02,
                         S=S,
                         T=2,
                         sample_percent=0.1,
                         base_classifier=LogisticRegression(**lr_config))
        sb.fit(labeled_feat=labeled_data,
               labels=labels)

        train_scores = sb.train_scores()
        test_scores = sb.decision_function(all_data)

        # We calculated the scores on the same dataset, so they must be the same
        assert_series_equal(train_scores, test_scores)

        test_probs = sb.predict_proba(all_data)
        test_predits = sb.predict(all_data)

        self.assertTrue(all((test_probs[:, 1] > 0.5) == (test_predits == 1)))


if __name__ == '__main__':
    unittest.main()
