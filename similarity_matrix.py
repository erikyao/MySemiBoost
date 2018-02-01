import numpy as np
from math import exp
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
    def __init__(self, X, sigma=1, dtype=None):
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

        self.sigma = sigma
        self.dtype = dtype

        sim_metric = partial(similarity, sigma=sigma)
        self.dense_matrix = pdist(X, sim_metric)

        if dtype is not None:
            """
            `copy=False` means:

            - if you are casting to the same dtype,
              this function makes no copy and return the caller itself;
            - if you are casting to a different dtype,
              a copy is always made and returned.
            """
            self.dense_matrix = self.dense_matrix.astype(dtype, copy=False)

        self.N = X.shape[0]

    def _convert_key_to_array(self, key):
        """
        No 3 kinds of keys are supported: int, slice, list, np.array.
        All kinds of keys are converted to lists for now.
        """
        if isinstance(key, int) or isinstance(key, np.integer):
            return np.array([key])
        elif isinstance(key, slice):
            indices = key.indices(self.N)
            return np.array([i for i in range(*indices)])
        elif isinstance(key, list):
            return np.array(key, dtype=np.integer)
        elif isinstance(key, np.ndarray):
            if key.dtype == bool:
                index = np.arange(self.N, dtype=np.integer)
                if len(index) != len(key):
                    raise ValueError("Mask array length {} does not match dimension N = {}".
                                     format(len(key), self.N))
                return index[key]
            else:
                return key
        else:
            raise TypeError("{} is not a integer, slice, list or np.array".foramt(key))

    def _getitem_int(self, i, j):
        """
        For each i and j (where i < j < N). The metric dist(u=X[i], v=X[j]) is stored
        in dense_matrix[(i+1)(j+1) - 1].

        :param i: int. Subscript of X, ranging from 0 to N-1
        :param j: int. Subscript of X, ranging from 0 to N-1
        :return:
        """
        if i < 0 or i >= self.N:
            raise ValueError("1st int index {} is out of bound.".format(i))
        if j < 0 or j >= self.N:
            raise ValueError("2nd int index {} is out of bound.".format(j))

        if i == j:
            yield 1
        else:
            if i < j:
                i, j = j, i  # swap i and j; ensure i > j

            # For indexing details, see http://yyao.info/scipy/2018/01/31/scipy-pdist-indexing
            # Python 3 int division will return a float
            dense_index = int(self.N*j - j*(j+1)/2 + i - j - 1)
            if dense_index >= len(self.dense_matrix):
                raise ValueError("Got i={} and j={}. Dense int index {} >= dense matrix length {}"
                                 .format(i, j, dense_index, len(self.dense_matrix)))
            yield self.dense_matrix[dense_index]

    def _getitem_array(self, array_i, array_j):
        for i in array_i:
            for j in array_j:
                yield from self._getitem_int(i, j)

    def __getitem__(self, key):
        """
        Call to `S[i, j]` will be interpreted as `type(S).__getitem__(S, (x, y))`
        """
        key_i, key_j = key  # unpack

        array_i = self._convert_key_to_array(key_i)
        array_j = self._convert_key_to_array(key_j)

        item_list = list(self._getitem_array(array_i, array_j))
        if len(item_list) == 1:
            # Got only one similarity value
            return item_list[0]
        else:
            # Got multiple similarity values. Put them into an array
            item_array = np.array(item_list, dtype=self.dtype)

            # Reshape this array if necessary
            if len(array_i) > 1 and len(array_j) > 1:
                item_array = item_array.reshape(len(array_i), len(array_j))

            return item_array

    def tenth_percentiles(self):
        """
        Percentiles of S are possible values for tuning sigma.
        """
        # 10th to 90th percentiles
        pct = np.percentile(self.dense_matrix, range(10, 100, 10))
        # 100th percentile is always 1
        pct = np.append(pct, 1)

        return pct

    def change_sigma(self, new_sigma):
        if self.sigma == new_sigma:
            return  # do nothing

        if self.sigma != 1:
            # S = S ** (cur_sigma ** 2)
            np.power(self.dense_matrix, self.sigma ** 2, out=self.dense_matrix)

        # S = S ** (1 / (new_sigma ** 2))
        np.power(self.dense_matrix, 1 / (new_sigma ** 2), out=self.dense_matrix)
