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
    def __init__(self, sigma=1, sample_proportion=0.1, T=20, base_classifier=None, random_state=None):
        # Similarity matrix is calculated individually
        # It's sigma may be different from our SemiBooster's sigma setting
        self.sigma = sigma

        self.sample_proportion = sample_proportion
        self.T = T
        self.base_classifier = base_classifier

        self.H = None

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

        C = X.shape[0] / Z.shape[0]

        self.H = self._run(X, y, Z, S, C)

        return self

    def _sample(self, array, weights, proportion):
        size = round(len(array) * proportion, ndigits=None)  # round to the nearest integer
        prob = weights / weights.sum()
        return self.rs.choice(array, size=size, replace=False, p=prob)

    def _run(self, X, y, Z, S, C):
        logger.debug("Running!")

        XZ = pd.concat([X, Z])
        H = Ensemble(t=0, index=XZ.index)

        for cur_round in range(1, self.T + 1):
            logger.debug("t = %d. Calculating pi and qi...", cur_round)

            # `DataFrame.apply(axis=1)` cannot access each row's index
            #   because it treats each row as a numpy object, not a Series.
            # (However you can use `lambda x: x.name` to access each row's name,
            #   which happens to be its index, when using `apply`)

            # `Index.map()` result does not have the original index
            # p_i_series = Z.index.map(lambda i: p_i(i, y, Z.index, S, H, C))
            # q_i_series = Z.index.map(lambda i: q_i(i, y, Z.index, S, H, C))

            p_q_array = np.array([pq_i(i, y, Z, S, H, C) for i in XZ.index.values])

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

            logger.debug("t = %d. Learning base clf...", cur_round)

            cur_clf = clone(self.base_classifier)
            cur_clf.fit(X_prime, y_prime)
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
                H.update(cur_clf, cur_alpha, h)
                logger.info("t = %d, alpha = %f. H updated.", cur_round, cur_alpha)
            else:
                print(cur_alpha)
                logger.info("t = %d, alpha = %f <= 0. Stop. Return H", cur_round, cur_alpha)
                break

        if len(H.clf_list) == 0 or len(H.alpha_list) == 0:
            raise NoBaseClassifierError("No base classifier learnt. Ensemble is empty")

        return H

    def train_scores(self):
        return self.H.scores

    def _zip_clf_and_alpha(self, t=None):
        if t is None:
            # Use all the clf's and alpha's
            return zip(self.H.clf_list, self.H.alpha_list)
        else:
            if t > len(self.H.clf_list):
                raise ParameterTError("Got t = {}. Only learnt {} base classifiers.".format(t, len(self.H.clf_list)))
            else:
                return zip(self.H.clf_list[0:t], self.H.alpha_list[0:t])

    def decision_function(self, X, t=None):
        if len(self.H.clf_list) == 0 or len(self.H.alpha_list) == 0:
            raise NoBaseClassifierError("No base classifier learnt. Cannot predict.")

        scores_list = [pd.Series(clf.predict(X), index=X.index) * alpha
                       for clf, alpha in self._zip_clf_and_alpha(t)]

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
        alpha_total = sum(self.H.alpha_list)

        # `predict_proba` returns an array of shape (n_samples, n_classes)
        # Cannot wrap this array into a pd.Series
        probs_list = [clf.predict_proba(X) * alpha
                      for clf, alpha in self._zip_clf_and_alpha(t)]
        probs_total = sum(probs_list)

        probs = np.divide(probs_total, alpha_total)

        # probs.columns = self.classes_

        return probs

    def accum_decision_function(self, X, t=None):
        if len(self.H.clf_list) == 0 or len(self.H.alpha_list) == 0:
            raise NoBaseClassifierError("No base classifier learnt. Cannot predict.")

        scores_list = [pd.Series(clf.predict(X), index=X.index) * alpha
                       for clf, alpha in self._zip_clf_and_alpha(t)]

        return accumulate(scores_list, pd.Series.add)

    def accum_predict(self, X, t=None):
        accum_scores = self.accum_decision_function(X, t)

        for scores in accum_scores:
            yield self._predict_from_scores(scores)

    def accum_predict_proba(self, X, t=None):
        probs_list = [clf.predict_proba(X) * alpha
                      for clf, alpha in self._zip_clf_and_alpha(t)]
        accum_probs = accumulate(probs_list)
        accum_alpha = accumulate(self.H.alpha_list)

        for alpha_sum, probs_sum in zip(accum_alpha, accum_probs):
            probs = np.divide(probs_sum, alpha_sum)

            # probs.columns = self.classes_

            yield probs
