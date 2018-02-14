import numpy as np
from math import log, exp
# import logging


# logger = logging.getLogger(__name__)

def I(a, b):
    return 1 if a == b else 0


# @profile
def pq_i(i, y, Z, S, H, C):
    """
    Calculate p_i and q_i at the same time to save computing resource
    """
    j_l = y.index
    j_u = Z.index

    # ----- Prepare for left parts ----- #
    # logger.debug("Fetching S[i, j_l.values] ...")
    S_i = S[i, j_l.values]

    # ----- Calculate p_i left ----- #
    # logger.debug("Calculating p_i left ...")
    I_j = (y.values == 1).astype(int)
    e_H = exp(-2 * H[i])
    p_i_left = sum(S_i * I_j * e_H)

    # ----- Calculate q_i left ----- #
    # logger.debug("Calculating q_i left ...")
    I_j = (y.values == -1).astype(int)
    e_H = 1 / e_H
    q_i_left = sum(S_i * I_j * e_H)

    # ----- Prepare for right parts ----- #
    # logger.debug("Fetching S[i, j_u.values] ...")
    S_i = S[i, j_u.values]

    # ----- Calculate p_i right ----- #
    # logger.debug("Calculating p_i right ...")
    '''
    Incredibly on my lab workstation:

    >>> %timeit -n100 -r100 s.apply(exp)
    100 loops, best of 100: 66.7 us per loop
    >>> %timeit -n100 -r100 np.exp(s)
    100 loops, best of 100: 24 us per loop

    >>> from pandas.testing import assert_series_equal
    >>> assert_series_equal(s.apply(exp), np.exp(s))
    '''
    # exp_H_diff = (H[j_u.values] - H[i]).apply(exp)
    exp_H_diff = np.exp(H[j_u.values] - H[i])
    p_i_right = 0.5 * C * sum(S_i * exp_H_diff)

    # ----- Calculate q_i right ----- #
    # logger.debug("Calculating q_i right ...")
    exp_H_diff = 1 / exp_H_diff
    q_i_right = 0.5 * C * sum(S_i * exp_H_diff)

    return p_i_left + p_i_right, q_i_left + q_i_right


def optimal_alpha(i_u, p, q, h):
    # i_u = y_u.index
    # p[i] = p_i(i, ...)
    numerator = sum(i_u.map(lambda i: p[i] * I(h[i], 1))) + sum(i_u.map(lambda i: q[i] * I(h[i], -1)))
    denominator = sum(i_u.map(lambda i: p[i] * I(h[i], -1))) + sum(i_u.map(lambda i: q[i] * I(h[i], 1)))

    return 0.25 * log(numerator / denominator)
