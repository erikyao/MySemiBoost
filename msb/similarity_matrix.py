import numpy as np
from math import exp
import gc
from scipy.spatial.distance import pdist, squareform
from functools import partial


def euclidean_distance(a, b):
    """
    The following code also works, but is slower.

    >>> from scipy.spatial import distance
    >>> distance.euclidean(a, b)

    """
    return np.linalg.norm(a - b, ord=2)


def similarity(a, b, sigma, dtype=None):
    numerator = -(euclidean_distance(a, b) ** 2)
    denominator = sigma ** 2

    sim = exp(numerator / denominator)

    if dtype is not None:
        sim = dtype(sim)

    return sim


class SimilarityMatrix:
    def __init__(self, sigma, dtype, full_matrix, N):
        self.sigma = sigma
        self.dtype = dtype
        self.full_matrix = full_matrix
        self.N = N

    @classmethod
    def compute(cls, X, sigma=1, dtype=None):
        """
        Previously, I tried to get an N*N full matrix by

        >>> sim_metric = partial(similarity, sigma=sigma)
        >>> matrix = squareform(pdist(dfm_ppr, sim_metric))
        >>> # `squareform` will expand `pdist` dense vector into a matrix.
        >>> # However, the diagonal defaults to 0.
        >>> np.fill_diagonal(matrix, 1)  # `fill_diagonal` is an in-place operation

        The following code also work, but is slower.

        >>> from sklearn.metrics.pairwise import pairwise_distances
        >>> # `fill_diagonal` not required
        >>> matrix = pairwise_distances(df, metric=sim_metric)

        Considering the vast memory space when N goes up to >= 10000,
            I'll just use the `pdist` dense matrix here.

        `__get_item__` needs to be taken good care of to handle
            access in the form of `S[i, j]`:
        """

        sim_metric = partial(similarity, sigma=sigma)
        dense_matrix = pdist(X, sim_metric)

        if dtype is not None:
            """
            `copy=False` means:

            - if you are casting to the same dtype,
              this function makes no copy and return the caller itself;
            - if you are casting to a different dtype,
              a copy is always made and returned.
            """
            dense_matrix = dense_matrix.astype(dtype, copy=False)

        full_matrix = squareform(dense_matrix)
        np.fill_diagonal(full_matrix, 1)

        # ----- GC dense_matrix ----- #
        del dense_matrix
        gc.collect()
        # --------------------------- #

        N = X.shape[0]

        return cls(sigma, dtype, full_matrix, N)

    @classmethod
    def load(cls, path, sigma, dtype, N):
        full_matrix = np.fromfile(path, dtype=dtype)

        # This is weird! Shape cannot be obtained from .dat file! It's alwasys a 1-D array of length N*N
        if (N * N) != len(full_matrix):
            raise ValueError("Got N = {}, not matching to full matrix length {}"
                             .format(N, len(full_matrix)))
        full_matrix.shape = (N, N)

        return cls(sigma, dtype, full_matrix, N)

    # def _convert_key_to_index(self, key):
    #     """
    #     Now 5 kinds of keys are supported: int, slice, list, range and np.array.
    #     """
    #     if isinstance(key, int) or isinstance(key, np.integer):
    #         return [key]
    #     elif isinstance(key, slice):
    #         indices = key.indices(self.N)
    #         return range(*indices)
    #     elif isinstance(key, list) or isinstance(key, range):
    #         return key
    #     elif isinstance(key, np.ndarray):
    #         if key.dtype == bool:
    #             index = np.arange(self.N, dtype=np.integer)
    #             if len(index) != len(key):
    #                 raise ValueError("Mask array length {} does not match dimension N = {}".
    #                                  format(len(key), self.N))
    #             return index[key]
    #         else:
    #             return key
    #     else:
    #         raise TypeError("{} is not a integer, slice, list, range or np.array".format(key))
    #
    # def _dense_index(self, i, j):
    #     # if i < j:
    #     #    i, j = j, i  # swap i and j; ensure i > j
    #     assert i >= 0 and i < self.N
    #     assert j >= 0 and j < self.N
    #     assert i > j
    #     # For indexing details, see http://yyao.info/scipy/2018/01/31/scipy-pdist-indexing
    #     # Python 3 int division will return a float
    #     dense_index = self.N*j - j*(j+1)//2 + i - j - 1
    #     assert dense_index < len(self.dense_matrix)
    #     return dense_index
    #
    # def _getitem_int(self, i, j):
    #     """
    #     :param i: int. Subscript of X, ranging from 0 to N-1
    #     :param j: int. Subscript of X, ranging from 0 to N-1
    #     :return:
    #     """
    #     # if i < 0 or i >= self.N:
    #     #     raise ValueError("1st int index {} is out of bound.".format(i))
    #     # if j < 0 or j >= self.N:
    #     #     raise ValueError("2nd int index {} is out of bound.".format(j))
    #
    #     if i == j:
    #         assert i >= 0 and i < self.N
    #         assert j >= 0 and j < self.N
    #         yield 1
    #     else:
    #         if i < j:
    #             dense_index = self._dense_index(j, i)
    #         else:
    #             dense_index = self._dense_index(i, j)
    #
    #         # if dense_index >= len(self.dense_matrix):
    #         #     raise ValueError("Got i={} and j={}. Dense int index {} >= dense matrix length {}"
    #         #                      .format(i, j, dense_index, len(self.dense_matrix)))
    #         yield self.dense_matrix[dense_index]
    #
    # def _getitem_index(self, index_i, index_j):
    #     for i in index_i:
    #         for j in index_j:
    #             yield from self._getitem_int(i, j)
    #
    # def __getitem__(self, key):
    #     """
    #     Call to `S[i, j]` will be interpreted as `type(S).__getitem__(S, (x, y))`
    #     """
    #     key_i, key_j = key  # unpack
    #
    #     index_i = self._convert_key_to_index(key_i)
    #     index_j = self._convert_key_to_index(key_j)
    #
    #     item_list = list(self._getitem_index(index_i, index_j))
    #     if len(item_list) == 1:
    #         # Got only one similarity value
    #         return item_list[0]
    #     else:
    #         # Got multiple similarity values. Put them into an array
    #         item_array = np.array(item_list, dtype=self.dtype)
    #
    #         # Reshape this array if necessary
    #         if len(index_i) > 1 and len(index_j) > 1:
    #             item_array = item_array.reshape(len(index_i), len(index_j))
    #
    #         return item_array

    def __getitem__(self, item):
        return self.full_matrix.__getitem__(item)

    def tenth_percentiles(self):
        """
        Percentiles of S are possible values for tuning sigma.
        """
        # 10th to 90th percentiles
        pct = np.percentile(self.full_matrix, range(10, 100, 10))

        return pct

    def change_sigma(self, new_sigma):
        if self.sigma == new_sigma:
            return  # do nothing

        if self.sigma != 1:
            # S = S ** (cur_sigma ** 2)
            np.power(self.full_matrix, self.sigma ** 2, out=self.full_matrix)

        # S = S ** (1 / (new_sigma ** 2))
        np.power(self.full_matrix, 1 / (new_sigma ** 2), out=self.full_matrix)

        np.fill_diagonal(self.full_matrix, 1)
