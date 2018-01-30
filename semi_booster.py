from os import path
import logging
import logging.config
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from .pseudo_label_util import p_i, q_i, alpha
from .sampling import sample_pseudo_weights


log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.ini')

logging.config.fileConfig(log_file_path)
logger = logging.getLogger()
logger.name = __name__


class Ensemble:
    def __init__(self, t, index):
        self.t = t
        self.clf_list = []   # the $h$ sequence which gives $h_1(x), h_2(x), ..., h_t(x)$
        self.alpha_list = []  # the \alpha sequence of $\alpha_1, \alpha_2, ..., \alpha_t$
        self.h_x_list = []

        self.scores = pd.Series(data=0, index=index)  # the values of $H(x)$

    def update(self, clf, alpha, h_x):
        self.t += 1
        self.clf_list.append(clf)
        self.alpha_list.append(alpha)
        self.h_x_list.append(h_x)

        if self.scores is None:
            self.scores = alpha * h_x
        else:
            self.scores += alpha * h_x

    def __getitem__(self, item):
        return self.scores[item]

    def __len__(self):
        return len(self.scores)


# `BaseEstimator` implemented: `set_params` method (used in GridSearch)
# `ClassifierMixin` implemented: default `score` method
class SemiBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, unlabeled_feat=pd.DataFrame(), sigma=1, S_factory=None,
                 sample_percent=0.1, T=20, base_classifier=None):
        self.labeled_feat = None
        self.labels = None
        self.unlabeled_feat = unlabeled_feat
        self.all_feat = None

        # By convention, attributes ending with '_' are estimated from data in scikit-learn.
        # Consequently they should not be initialized in the constructor of an estimator but in the `fit` method.
        # self.classes_ = None

        self.C = None

        self.sigma = sigma
        self.S_factory = S_factory
        self.S = None

        self.sample_percent = sample_percent
        self.T = T
        self.base_classifier = base_classifier

        self.H = None

    def fit(self, labeled_feat, labels):
        self.labeled_feat = labeled_feat
        self.labels = labels
        # self.unlabeled_feat = unlabeled_feat
        self.all_feat = pd.concat([labeled_feat, self.unlabeled_feat])

        # `np.unique` returns SORTED unique values
        self.classes_ = np.unique(labels)

        # Only evaluate S at the 1st time or when self.sigma changes
        if (self.S is None) or (self.sigma != self.S_factory.last_sigma):
            self.S = self.S_factory.produce(self.sigma)

        self.C = len(labeled_feat.index) / len(self.unlabeled_feat.index)

        self.H = self._run()

        return self

    def _run(self):
        H = Ensemble(t=0, index=self.all_feat.index)

        for cur_round in range(1, self.T + 1):
            # `DataFrame.apply(axis=1)` cannot access each row's index
            #   because it treats each row as a numpy object, not a Series.
            # (However you can use `lambda x: x.name` to access each row's name,
            #   which happens to be its index, when using `apply`)

            # `Index.map()` cannot keep index
            p_i_series = self.unlabeled_feat.index.map(lambda i:
                                                       p_i(i, self.labels, self.unlabeled_feat.index,
                                                           self.S, H, self.C))
            p_i_series = pd.Series(p_i_series, index=self.unlabeled_feat.index)
            q_i_series = self.unlabeled_feat.index.map(lambda i:
                                                       q_i(i, self.labels, self.unlabeled_feat.index,
                                                           self.S, H, self.C))
            q_i_series = pd.Series(q_i_series, index=self.unlabeled_feat.index)

            # `np.sign` may return floats
            pseudo_labels = pd.Series(np.sign(p_i_series - q_i_series).astype(int), index=self.unlabeled_feat.index)
            pseudo_weights = pd.Series(abs(p_i_series - q_i_series), index=self.unlabeled_feat.index)

            sampled_index = sample_pseudo_weights(pseudo_weights, self.sample_percent)
            sampled_unlabeled_data = self.unlabeled_feat.loc[sampled_index, ]
            sampled_pseudo_labels = pseudo_labels[sampled_index]

            cur_clf = clone(self.base_classifier)
            X = pd.concat([self.labeled_feat, sampled_unlabeled_data])
            y = pd.concat([self.labels, sampled_pseudo_labels])
            cur_clf.fit(X, y)
            # `cur_clf.predict` cannot keep index
            h = pd.Series(cur_clf.predict(self.all_feat), index=self.all_feat.index)
            cur_alpha = alpha(sampled_index, p_i_series[sampled_index], q_i_series[sampled_index], h[sampled_index])

            if cur_alpha > 0.0:
                logger.info("t = %d, alpha = %f", cur_round, cur_alpha)
                H.update(cur_clf, cur_alpha, h)
            else:
                logger.info("t = %d, alpha = %f <= 0. Stop.", cur_round, cur_alpha)
                break

        return H

    def train_scores(self):
        return self.H.scores

    def decision_function(self, X):
        scores_list = [pd.Series(clf.predict(X), index=X.index) * alpha
                      for clf, alpha in zip(self.H.clf_list, self.H.alpha_list)]
        scores = sum(scores_list)

        return scores

    def predict(self, X):
        # See https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/linear_model/base.py#L311
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        alpha_total = sum(self.H.alpha_list)

        probs_lst = [clf.predict_proba(X) * alpha
                     for clf, alpha in zip(self.H.clf_list, self.H.alpha_list)]
        probs_total = sum(probs_lst)

        probs = np.divide(probs_total, alpha_total)

        # probs.columns = self.classes_

        return probs

