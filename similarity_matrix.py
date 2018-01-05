import pandas as pd
import numpy as np
from math import exp
from scipy.spatial.distance import pdist, squareform
from functools import partial


def l2_distance(a, b):
    """
    The following code also works, but is slower.

        from scipy.spatial import distance

        return distance.euclidean(a,b)

    """
    return np.linalg.norm(a - b, ord=2)


def similarity(a, b, sigma):
    numerator = -(l2_distance(a, b) ** 2)
    denominator = sigma ** 2

    return exp(numerator / denominator)


def similarity_matrix(dfm, sigma, preprocessor):
    """
    The following code also work, but is slower.

        from sklearn.metrics.pairwise import pairwise_distances

        # No call to `fill_diagonal` required
        matrix = pairwise_distances(df, metric=partial(similarity, sigma=sigma))

    """
    if preprocessor is None:
        dfm_ppr = dfm
    else:
        dfm_ppr = pd.DataFrame(preprocessor.fit_transform(dfm))

    matrix = squareform(pdist(dfm_ppr, metric=partial(similarity, sigma=sigma)))
    # `fill_diagonal` is an in-place operation
    # `pdist` returns a dense vector of pair-wise distances (`similarity` actually here)
    # `squareform` will expand this dense vector into a matrix. However, the diagonal defaults to 0
    #   We need to fill diagonal with 1 (self-distance is 0 while self-similarity is 1)
    np.fill_diagonal(matrix, 1)

    # encapsulate into a DataFrame
    matrix = pd.DataFrame(matrix)
    matrix.index = dfm.index
    matrix.columns = dfm.index

    return matrix


def similarity_matrix_percentiles(S):
    """
    Percentiles of S are possible values for tuning sigma.

    :param S:
    :return:
    """

    # 10th to 90th percentiles
    pct = np.percentile(S, range(10, 100, 10))
    # 100th percentile is always 1
    pct = np.append(pct, 1)

    return pct


def change_sigma(S, cur_sigma, new_sigma):
    if cur_sigma == new_sigma:
        return S.copy()

    # Note that `x ** y ** z` is actually `x ** (y ** z)`
    # Take care if you want to refactor this part

    if cur_sigma != 1:
        # `S ** n` is NOT a inplace operation;
        # it will always return a new DataFrame without changing S
        S = S ** (cur_sigma ** 2)

    S = S ** (1 / (new_sigma ** 2))

    np.fill_diagonal(S.values, 1)

    return S


class SimilarityMatrixFactory:
    base_sigma = 1

    def __init__(self, feat_dfm, preprocessor):
        self.base_matrix = similarity_matrix(feat_dfm, SimilarityMatrixFactory.base_sigma, preprocessor)
        self.last_sigma = None

    def produce(self, sigma):
        self.last_sigma = sigma
        return change_sigma(self.base_matrix, SimilarityMatrixFactory.base_sigma, sigma)