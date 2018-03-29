from __future__ import print_function
from __future__ import division

import warnings
import numbers
import time
import itertools

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable,  safe_indexing
from sklearn.utils.deprecation import DeprecationDict
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv


def _chunks(seq, size):
    """Yield successive chunks of `size` from `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def _split_dict(decision_scores, chunk_size):
    """
    split `decision_scores`, a dicctionary whose values are sequences of the same length `m` into m/n sub-dictionaries,
    where `n` is the length of the values in those sub-dictionaries

    :param decision_scores: a dictionary whose values are sequences of the same length
    :param chunk_size: length of sequences in sub-dictionaries' values
    :return: a generator of sub dictionaries
    """

    """
    E.g. a dictionary with the following keys and values, is split with `chunk_size=1` as below

    | key | value | value-chunk-1 | value-chunk-2 |
    |-----|-------|---------------|---------------|
    | k1  | (1,2) |       1       |       2       |
    | k2  | (3,4) |       3       |       4       |

    2 sub-dictionaries will be yielded:

    | key | value |  | key | value |
    |-----|-------|  |-----|-------|
    | k1  |   1   |  | k1  |   2   |
    | k2  |   3   |  | k2  |   4   |
    """
    value_chunks = zip(*(_chunks(value_seq, chunk_size) for value_seq in decision_scores.values()))
    keys = decision_scores.keys()
    for chunk in value_chunks:
        yield dict(zip(keys, chunk))


def repeated_cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None,
                            n_jobs=1, n_reps=1, verbose=0, fit_params=None,
                            pre_dispatch='2*n_jobs', return_train_score="warn"):
    if len(cv) != n_reps:
        raise ValueError("Set n_reps = {}. Got only {} cross validators.".format(n_reps, len(cv)))

    n_folds = np.unique([cross_validator.get_n_splits() for cross_validator in cv])
    if len(n_folds) != 1:
        raise ValueError("Cross validators are not unified in fold number: {}".format(n_folds))
    n_folds = n_folds[0]

    """Evaluate metric(s) by cross-validation and also record fit/score times.

        Read more in the :ref:`User Guide <multimetric_cross_validation>`.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : array-like
            The data to fit. Can be for example a list, or an array.

        y : array-like, optional, default: None
            The target variable to try to predict in the case of
            supervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        scoring : string, callable, list/tuple, dict or None, default: None
            A single string (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique) strings
            or a dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a single
            value. Metric functions returning a list/array of values can be wrapped
            into multiple scorers that return one value each.

            See :ref:`multimetric_grid_search` for an example.

            If None, the estimator's default scorer (if available) is used.

        cv : array-like, a collection of cross-validation generators, with length n_reps

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        n_jobs : integer, optional
            The number of CPUs to use to do the computation. -1 means
            'all CPUs'.

        verbose : integer, optional
            The verbosity level.

        fit_params : dict, optional
            Parameters to pass to the fit method of the estimator.

        pre_dispatch : int, or string, optional
            Controls the number of jobs that get dispatched during parallel
            execution. Reducing this number can be useful to avoid an
            explosion of memory consumption when more jobs get dispatched
            than CPUs can process. This parameter can be:

                - None, in which case all the jobs are immediately
                  created and spawned. Use this for lightweight and
                  fast-running jobs, to avoid delays due to on-demand
                  spawning of the jobs

                - An int, giving the exact number of total jobs that are
                  spawned

                - A string, giving an expression as a function of n_jobs,
                  as in '2*n_jobs'

        return_train_score : boolean, optional
            Whether to include train decision_scores.

            Current default is ``'warn'``, which behaves as ``True`` in addition
            to raising a warning when a training score is looked up.
            That default will be changed to ``False`` in 0.21.
            Computing training decision_scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the decision_scores on the training set can be computationally
            expensive and is not strictly required to select the parameters that
            yield the best generalization performance.

        Returns
        -------
        repeated_decision_scores : dict of `decision_scores` dicts, of shape=(n_reps,)
    """
    X, y, groups = indexable(X, y, groups)

    # cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    # ---------------------- My Hack ----------------------- #
    # 1) Set parameter `error_score=-1` to `_fit_and_score`  #
    # 2) Created an argument `return_estimator` to           #
    #    `_fit_and_score`                                    #
    # ------------------------------------------------------ #
    tasks = [[
                 delayed(_fit_and_score)(clone(estimator), X, y, scorers, train, test, verbose, None,
                                         fit_params, return_train_score=return_train_score,
                                         return_times=True, return_estimator=True, error_score=-1)
                 for train, test in cross_validator.split(X, y, groups)
             ] for cross_validator in cv]
    # Flatten this list of lists into a simple list
    tasks = itertools.chain.from_iterable(tasks)
    scores = parallel(tasks)

    if return_train_score:
        train_scores, test_scores, fit_times, score_times, estimators = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times, estimators = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)
    ret['estimator'] = list(estimators)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training decision_scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    """
    Now `ret` is a dictionary whose values are all sequences of length `n_folds * n_reps`.
    Split it into `n_reps` sub-dictionaries whose values are of length `n_folds`
    """
    rep_rets = list(_split_dict(ret, chunk_size=n_folds))

    assert len(rep_rets) == n_reps

    for i in range(0, n_reps):
        rep_rets[i]["cross_validator"] = cv[i]

    result = dict(zip(range(0, n_reps), rep_rets))

    return result


def cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None,
                   n_jobs=1, verbose=0, fit_params=None,
                   pre_dispatch='2*n_jobs', return_train_score="warn"):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be for example a list, or an array.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cross_validators are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : boolean, optional
        Whether to include train decision_scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training decision_scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the decision_scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Returns
    -------
    decision_scores : dict of float arrays of shape=(n_splits,)
        Array of results of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test decision_scores on each cross_validators split.
            ``train_score``
                The score array for train decision_scores on each cross_validators split.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cross_validators split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cross_validators split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                A list of estimator objects, one for each training dataset.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics.scorer import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, return_train_score=False)
    >>> sorted(cv_results.keys())                         # doctest: +ELLIPSIS
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']    # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([ 0.33...,  0.08...,  0.03...])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> decision_scores = cross_validate(lasso, X, y,
    ...                         scoring=('r2', 'neg_mean_squared_error'))
    >>> print(decision_scores['test_neg_mean_squared_error'])      # doctest: +ELLIPSIS
    [-3635.5... -3573.3... -6114.7...]
    >>> print(decision_scores['train_r2'])                         # doctest: +ELLIPSIS
    [ 0.28...  0.39...  0.22...]

    See Also
    ---------
    :func:`sklearn.model_selection.cross_val_score`:
        Run cross-validation for single metric evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    # ---------------------- My Hack ----------------------- #
    # 1) Set parameter `error_score=-1` to `_fit_and_score`  #
    # 2) Created an argument `return_estimator` to           #
    #    `_fit_and_score`                                    #
    # ------------------------------------------------------ #
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True, return_estimator=True, error_score=-1)
        for train, test in cv.split(X, y, groups))

    if return_train_score:
        train_scores, test_scores, fit_times, score_times, estimators = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times, estimators = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)
    ret['estimator'] = list(estimators)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training decision_scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    ret['cross_validator'] = cv

    return ret


def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False, error_score='raise'):
    """Fit estimator and compute decision_scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    return_estimator : boolean, optional, default: False
        Whether to return the fitted estimators.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.

    estimator : object
        The fitted estimator
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                   [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                        [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


def _score(estimator, X_test, y_test, scorer, is_multimetric=False):
    """Compute the score(s) of an estimator on a given test set.

    Will return a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:
        return _multimetric_score(estimator, X_test, y_test, scorer)
    else:
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%r)"
                             % (str(score), type(score), scorer))
    return score


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the decision_scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> decision_scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(decision_scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    out = {}
    for key in scores[0]:
        out[key] = np.asarray([score[key] for score in scores])
    return out


def _index_param_value(X, v, indices):
    """Private helper function for parameter value indexing."""
    if not _is_arraylike(v) or _num_samples(v) != _num_samples(X):
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)


def _multimetric_score(estimator, X_test, y_test, scorers):
    """Return a dict of score for multimetric scoring"""
    scores = {}

    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, 'item'):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError("scoring must return a number, got %s (%s) "
                             "instead. (scorer=%s)"
                             % (str(score), type(score), name))
    return scores