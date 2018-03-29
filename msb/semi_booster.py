import math
from itertools import accumulate
import logging
import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from .pseudo_label_util import pq_i, optimal_alpha

logger = logging.getLogger(__name__)
logger.debug("Imported!")


class NoBaseClassifierError(Exception):
    """
    When there is no base classifier learnt in a Ensemble
    """


class LabelError(Exception):
    """
    Labels must be either -1 or 1; otherwise this error should be raised.
    """


class ParameterTError(Exception):
    """
    When you specified a round number `t` greater than the number of the base classifiers learnt
    """


class Ensemble:
    def __init__(self, dummy_alpha, dummy_clf):
        self.t = 0

        if dummy_alpha == "class-balance":
            logger.info("Got dummy_alpha = 'class-balance'; value will be fixed in `dummy_fit`")
        elif dummy_alpha <= 0:
            raise ValueError("Numeric dummy_alpha must be positive. Got {}".format(dummy_alpha))

        if dummy_clf is None:
            logger.warn("Not using DummyClassifier. Set dummy_alpha = 0. Ignore passed value {}".format(dummy_alpha))

            self.dummy_alpha = 0  # $\alpha_0$
            self.dummy_clf = None  # $h_0(x)$
        else:
            self.dummy_alpha = dummy_alpha
            self.dummy_clf = dummy_clf

        self.clf_list = []   # sequence of $h_1(x), h_2(x), ..., h_t(x)$
        self.alpha_list = []  # sequence of $\alpha_1, \alpha_2, ..., \alpha_t$

        self.scores = None  # the values of $H(x)$

    def update(self, clf, alpha, h_x):
        self.t += 1
        self.clf_list.append(clf)
        self.alpha_list.append(alpha)

        if self.scores is None:
            self.scores = alpha * h_x
        else:
            self.scores += alpha * h_x

    def dummy_fit(self, X, y):
        if self.dummy_clf is not None:
            self.dummy_clf.fit(X, y)

            if self.dummy_alpha == "class-balance":
                self.dummy_alpha = abs(math.log(sum(y == 1) / sum(y == -1)) / 4)
                logger.info("Got dummy_alpha = 'class-balance'; reset it to {}".format(self.dummy_alpha))
        else:
            logger.info("No dummy classifier to train.")

    def dummy_init_scores(self, X):
        if self.dummy_clf is not None:
            self.scores = pd.Series(data=self.dummy_clf.predict(X) * self.dummy_alpha, index=X.index)
        else:
            self.scores = pd.Series(data=0, index=X.index)
            logger.info("No dummy classifier; set `scores` to 0")

    def __getitem__(self, item):
        return self.scores[item]

    def __len__(self):
        return len(self.scores)


# `BaseEstimator` implemented: `set_params` method (used in GridSearch)
# `ClassifierMixin` implemented: default `score` method
class SemiBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=1, sample_proportion=0.1, C="XZ-ratio", T=20, base_classifier=None,
                 dummy_alpha="class-balance", dummy_classifier=None, random_state=None):
        # Similarity matrix is calculated individually
        # It's sigma may be different from our SemiBooster's sigma setting
        self.sigma = sigma

        self.sample_proportion = sample_proportion
        self.C = C
        self.T = T
        self.base_classifier = base_classifier

        self.dummy_alpha = dummy_alpha
        self.dummy_classifier = dummy_classifier
        self.H = Ensemble(dummy_alpha, dummy_classifier)

        self.random_state = random_state
        self.rs = RandomState(random_state)

    def fit(self, X, y, Z, S):
        """

        :param X: the labeled features
        :param y: the labels
        :param Z: the unlabeled features
        :param S: the similarity matrix
        :return:
        """

        # By convention, attributes ending with '_' are estimated from data in scikit-learn.
        # Consequently they should not be initialized in the constructor of an estimator but in the `fit` method.
        # `np.unique` returns SORTED unique values
        self.classes_ = np.unique(y)

        pos_label_included = 1 in self.classes_
        neg_label_included = -1 in self.classes_
        if len(self.classes_) != 2 or not pos_label_included or not neg_label_included:
            err_msg = "Got invalid labels {}".format(self.classes_)
            logger.error(err_msg)
            raise LabelError(err_msg)

        # Reset S if self.sigma changes
        # When tuning for optimal sigma, disable parallelism because S is a singleton
        if S.sigma != self.sigma or not math.isclose(S.sigma, self.sigma, rel_tol=1e-6):
            logger.warn("S's sigma changing! {} => {}".format(S.sigma, self.sigma))
            S.change_sigma(new_sigma=self.sigma)

        labeled_ratio = X.shape[0] / (X.shape[0] + Z.shape[0])
        if self.sample_proportion <= labeled_ratio:
            logger.warn("{:.1%} data are labeled. Got sampling size {:.1%}. Samples may be all labeled."
                        .format(labeled_ratio, self.sample_proportion))
        else:
            logger.info("{:.1%} data are labeled. Got sampling size {:.1%}"
                        .format(labeled_ratio, self.sample_proportion))

        if self.C == "XZ-ratio":
            C = X.shape[0] / Z.shape[0]
        else:
            C = self.C

        self._run(X, y, Z, S, C)

        return self

    def _sample(self, array, weights, proportion):
        size = round(len(array) * proportion, ndigits=None)  # round to the nearest integer
        prob = weights / weights.sum()
        return self.rs.choice(array, size=size, replace=False, p=prob)

    def _run(self, X, y, Z, S, C):
        logger.debug("Running!")

        XZ = pd.concat([X, Z], axis=0, ignore_index=False)

        self.H.dummy_fit(X, y)
        self.H.dummy_init_scores(XZ)

        for cur_round in range(1, self.T + 1):
            logger.debug("t = %d. Calculating pi and qi...", cur_round)

            # `DataFrame.apply(axis=1)` cannot access each row's index
            #   because it treats each row as a numpy object, not a Series.
            # (However you can use `lambda x: x.name` to access each row's name,
            #   which happens to be its index, when using `apply`)

            # `Index.map()` result does not have the original index
            # p_i_series = Z.index.map(lambda i: p_i(i, y, Z.index, S, H, C))
            # q_i_series = Z.index.map(lambda i: q_i(i, y, Z.index, S, H, C))

            p_q_array = np.array([pq_i(i, y, Z, S, self.H, C) for i in XZ.index.values])

            p_i_series = pd.Series(p_q_array[:, 0], index=XZ.index)
            q_i_series = pd.Series(p_q_array[:, 1], index=XZ.index)

            logger.debug("t = %d. Pseduo labeling...", cur_round)

            # `np.sign` may return floats
            # It's possible that p_i == q_i and then its pseudo label is 0
            # When this happens, this example cannot be sampled out because its prob is also 0
            pseudo_labels = pd.Series(np.sign(p_i_series - q_i_series).astype(int), index=XZ.index)
            pseudo_weights = pd.Series(abs(p_i_series - q_i_series), index=XZ.index)

            logger.debug("t = %d. Sampling...", cur_round)

            sampled_index = self._sample(XZ.index, pseudo_weights, self.sample_proportion)

            X_prime = XZ.loc[sampled_index, ]
            y_prime = pseudo_labels.loc[sampled_index]
            w_prime = pseudo_weights.loc[sampled_index]

            logger.debug("t = %d. Learning base clf...", cur_round)

            cur_clf = clone(self.base_classifier)
            cur_clf.fit(X_prime, y_prime, sample_weight=w_prime)
            # `cur_clf.predict` cannot keep index
            h = pd.Series(cur_clf.predict(XZ), index=XZ.index)

            try:
                logger.debug("t = %d. Calculating alpha...", cur_round)
                cur_alpha = optimal_alpha(Z.index,
                                          p_i_series.loc[Z.index],
                                          q_i_series.loc[Z.index],
                                          h.loc[Z.index])
            except ZeroDivisionError:
                logger.error("t = %d. ZeroDivisionError when calculating alpha. Stop. Return H", cur_round)
                break

            if cur_alpha > 0.0:
                self.H.update(cur_clf, cur_alpha, h)
                logger.info("t = %d, alpha = %f. H updated.", cur_round, cur_alpha)
            else:
                print(cur_alpha)
                logger.info("t = %d, alpha = %f <= 0. Stop. Return H", cur_round, cur_alpha)
                break

        if not self.has_learnt_base_classifier():
            raise NoBaseClassifierError("No base classifier learnt. Ensemble is empty")

        return self.H

    def train_scores(self):
        return self.H.scores

    def _list_clf_and_alpha(self, t=None):
        if (t is not None) and (t > len(self.H.clf_list)):
            # Grammatically, it's OK to slice like `clf_list[0:t]` when `t` is greater than the list length,
            # but in this case I assume it's an error
            raise ParameterTError("Got t = {}. Only learnt {} base classifiers.".format(t, len(self.H.clf_list)))

        if self.has_dummy_classifier():
            # Slicing with `t=None` is allowed
            # `l[0:None]` is equivalent to `l[0:]`
            clf_list = [self.H.dummy_clf] + self.H.clf_list[0:t]
            alpha_list = [self.H.dummy_alpha] + self.H.alpha_list[0:t]
        else:
            clf_list = self.H.clf_list[0:t]
            alpha_list = self.H.alpha_list[0:t]

        return clf_list, alpha_list

    def decision_function(self, X, t=None):
        if not self.has_learnt_base_classifier():
            raise NoBaseClassifierError("No base classifier learnt. Cannot predict.")

        clf_list, alpha_list = self._list_clf_and_alpha(t)

        scores_list = [pd.Series(clf.predict(X), index=X.index) * alpha
                       for clf, alpha in zip(clf_list, alpha_list)]

        scores = sum(scores_list)

        return scores

    def _predict_from_scores(self, scores):
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict(self, X, t=None):
        # See https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/linear_model/base.py#L311
        scores = self.decision_function(X, t)
        return self._predict_from_scores(scores)

    def predict_proba(self, X, t=None):
        clf_list, alpha_list = self._list_clf_and_alpha(t)

        alpha_total = sum(alpha_list)

        # `predict_proba` returns an array of shape (n_samples, n_classes)
        # Cannot wrap this array into a pd.Series
        probs_list = [clf.predict_proba(X) * alpha
                      for clf, alpha in zip(clf_list, alpha_list)]
        probs_total = sum(probs_list)

        probs = np.divide(probs_total, alpha_total)

        # probs.columns = self.classes_

        return probs

    def accum_decision_function(self, X, t=None):
        if not self.has_learnt_base_classifier():
            raise NoBaseClassifierError("No base classifier learnt. Cannot predict.")

        clf_list, alpha_list = self._list_clf_and_alpha(t)

        scores_list = [pd.Series(clf.predict(X), index=X.index) * alpha
                       for clf, alpha in zip(clf_list, alpha_list)]

        acc_scores = accumulate(scores_list, pd.Series.add)

        if self.has_dummy_classifier():
            next(acc_scores)  # skip the dummy scores

        return acc_scores

    def stepwise_predict(self, X, t=None):
        accum_scores = self.accum_decision_function(X, t)

        for scores in accum_scores:
            yield self._predict_from_scores(scores)

    def stepwise_predict_proba(self, X, t=None):
        if not self.has_learnt_base_classifier():
            raise NoBaseClassifierError("No base classifier learnt. Cannot predict.")

        clf_list, alpha_list = self._list_clf_and_alpha(t)

        probs_list = [clf.predict_proba(X) * alpha
                      for clf, alpha in zip(clf_list, alpha_list)]
        accum_probs = accumulate(probs_list)
        accum_alpha = accumulate(alpha_list)

        if self.has_dummy_classifier():
            # skip the dummies
            next(accum_probs)
            next(accum_alpha)

        for alpha_sum, probs_sum in zip(accum_alpha, accum_probs):
            probs = np.divide(probs_sum, alpha_sum)

            # probs.columns = self.classes_

            yield probs

    def has_learnt_base_classifier(self):
        assert len(self.H.clf_list) == len(self.H.alpha_list)
        return (len(self.H.clf_list) > 0) and (len(self.H.alpha_list) > 0)

    def has_dummy_classifier(self):
        return self.H.dummy_clf is not None
