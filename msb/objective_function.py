from itertools import product
from math import exp


def F_u(y_u, S):
    ij_pairs = list(product(y_u.index, repeat=2))

    def y_diff(ij_pair):
        i = ij_pair[0]
        j = ij_pair[1]
        return y_u[i] - y_u[j]

    return sum([S.loc[ij_pair] * exp(y_diff(ij_pair)) for ij_pair in ij_pairs])


def F_l(y_l, y_u, S):
    ij_pairs = list(product(y_l.index, y_u.index))

    def y_prod(ij_pair):
        i = ij_pair[0]
        j = ij_pair[1]
        return y_l[i] * y_u[j]

    return sum([S.loc[ij_pair] * exp(-2 * y_prod(ij_pair)) for ij_pair in ij_pairs])


def F(y_l, y_u, S, C):
    return F_l(y_l, y_u, S) + C * F_u(y_u, S)
