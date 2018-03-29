from enum import Enum
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from locus_sampling.scoring import avg_rank_func


class Metric(Enum):
    ACC = "accuracy"
    AUROC = "AUROC"
    AUPVR = "AUPVR"
    AVGRANK = "AVGRANK"
    AVGRANK_EX = "AVGRANK (Loners Excluded)"


class PerformanceAgent:
    def __init__(self, y, decision_scores, pred, proba, groups):
        """

        :param y: The ground true labels
        :param decision_scores: The return values from `clf.decision_function`
        :param pred: The predicted labels
        :param proba: The predicted probabilities
        :param groups: The group id per example
        """
        self.y = y
        self.decision_scores = decision_scores
        self.pred = pred
        self.proba = proba
        self.groups = groups

    def measure(self, metric):
        if metric == Metric.ACC:
            return accuracy_score(self.y, self.pred)
        elif metric == Metric.AUROC:
            return roc_auc_score(self.y, self.decision_scores)
        elif metric == Metric.AUPVR:
            return average_precision_score(self.y, self.decision_scores)
        elif metric == Metric.AVGRANK:
            return avg_rank_func(self.y, self.proba, groups=self.groups)
        elif metric == Metric.AVGRANK_EX:
            return avg_rank_func(self.y, self.proba, groups=self.groups, exclude_loners=True)
        else:
            raise NotImplementedError("Metric '{}' not implemented yet!".format(metric))

    def measure_all(self, metrics):
        return tuple(self.measure(metric) for metric in metrics)


class PerformanceDataFrameBuilder:
    @staticmethod
    def _build_single_fold_performance_df(X, y, g_X, estimator, metrics):
        raise NotImplementedError("This is an abstract method!")

    @classmethod
    def _build_all_fold_performance_df(cls, X, y, g_X, cross_validator, estimators, metrics,
                                       return_train_performance=False):
        for estimator, (train_index, test_index) in zip(estimators, cross_validator.split(X, y, groups=g_X)):
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            test_performance = cls._build_single_fold_performance_df(X_test, y_test, g_X, estimator, metrics)
            test_performance = test_performance.assign(dataset="test")

            if return_train_performance:
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]

                train_performance = cls._build_single_fold_performance_df(X_train, y_train, g_X, estimator, metrics)
                train_performance = train_performance.assign(dataset="train")

                yield pd.concat([train_performance, test_performance], axis=0, ignore_index=True)
            else:
                yield test_performance

    @classmethod
    def build_single_cv_performance_df(cls, X, y, g_X, cross_validator, cv_end_product, metrics,
                                       return_train_performance=False):
        try:
            estimators = cv_end_product[0]["estimator"]
        except KeyError:
            # For back-compatibility
            estimators = cv_end_product["estimator"]

        all_fold_performance = list(cls._build_all_fold_performance_df(X, y, g_X, cross_validator, estimators,
                                                                       metrics, return_train_performance))

        for fold_num in range(0, len(all_fold_performance)):
            all_fold_performance[fold_num] = all_fold_performance[fold_num].assign(fold=fold_num + 1)

        single_cv_performance_df = pd.concat(all_fold_performance, axis=0, ignore_index=True)

        return single_cv_performance_df

    @classmethod
    def build_repeated_cv_performance_df(cls, X, y, g_X, cross_validators, n_reps, cv_end_product,
                                         metrics, return_train_performance=False):
        if len(cross_validators) != n_reps:
            raise ValueError("Set to repeat {} time(s). Got only {} cross validators: {}".
                             format(n_reps, len(cross_validators), cross_validators))

        for rep_num in range(0, n_reps):
            if rep_num not in cv_end_product: # check if key exists
                raise ValueError("Repeat number {} not in `cv_end_product` keys: {}".
                                 format(rep_num, cv_end_product.keys()))

        repeated_cv_performance = [None] * n_reps
        for rep_num in range(0, n_reps):
            single_cv_end_product = cv_end_product[rep_num]
            cross_validator = cross_validators[rep_num]

            single_cv_performance_df = cls.build_single_cv_performance_df(X, y, g_X, cross_validator,
                                                                          single_cv_end_product,
                                                                          metrics, return_train_performance)

            repeated_cv_performance[rep_num] = single_cv_performance_df.assign(rep=rep_num)

        repeated_cv_performance_df = pd.concat(repeated_cv_performance, axis=0, ignore_index=True)

        return repeated_cv_performance_df


class BaseClassifierPerformanceDataFrameBuilder(PerformanceDataFrameBuilder):
    @staticmethod
    def _build_single_fold_performance_df(X, y, g_X, estimator, metrics):
        # Some base classifier may not implement `decision_function`
        # See https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/scorer.py#L186
        try:
            decision_scores = estimator.decision_function(X)
        except (NotImplementedError, AttributeError):
            decision_scores = estimator.predict_proba(X)[:, 1]
        pred = estimator.predict(X)
        proba = estimator.predict_proba(X)

        pa = PerformanceAgent(y, decision_scores, pred, proba, g_X)
        performance = pa.measure_all(metrics)
        metric_names = [metric.value for metric in metrics]

        single_fold_performance_df = pd.DataFrame(dict(value=performance,
                                                       metric=metric_names))
        single_fold_performance_df = single_fold_performance_df.assign(t=0)

        return single_fold_performance_df


class SemiBoostPerformanceDataFrameBuilder(PerformanceDataFrameBuilder):
    @staticmethod
    def _build_single_fold_performance_df(X, y, g_X, estimator, metrics):
        # Accumulated decision_scores, predictions and probas in current fold
        accum_decision_scores = list(estimator.accum_decision_function(X))
        accum_pred = list(estimator.stepwise_predict(X))
        accum_proba = list(estimator.stepwise_predict_proba(X))

        # Per-round measurements in current fold
        performance = []
        for decision_scores, pred, proba in zip(accum_decision_scores, accum_pred, accum_proba):
            pa = PerformanceAgent(y, decision_scores, pred, proba, g_X)
            performance.append(pa.measure_all(metrics))

        metric_names = [metric.value for metric in metrics]

        single_fold_performance_df = pd.DataFrame(performance)
        single_fold_performance_df.columns = metric_names

        single_fold_performance_df = single_fold_performance_df.assign(t=range(1, estimator.T + 1))

        single_fold_performance_df = pd.melt(single_fold_performance_df, id_vars='t',
                                             value_vars=metric_names,
                                             var_name='metric', value_name='value')

        return single_fold_performance_df
