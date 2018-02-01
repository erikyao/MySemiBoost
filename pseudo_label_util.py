from math import log, exp
from functools import partial


def I(a, b):
    return 1 if a == b else 0


def p_i(i, y_l, j_u, S, H, C):
    # i = x_i.index
    j_l = y_l.index
    # j_u = y_u.index

    S_i = S[i, j_l.values]
    I_j = y_l.map(partial(I, b=1))
    left = sum(S_i * I_j * exp(-2 * H[i]))

    S_i = S[i, j_u.values]
    exp_H_diff = (H[j_u.values] - H[i]).apply(exp)
    right = 0.5 * C * sum(S_i * exp_H_diff)

    return left + right

# def p_i(i, y_l, j_u, S, H, C):
#     # i = x_i.index
#     j_l = y_l.index
#     # j_u = y_u.index
#
#     left = sum(j_l.map(lambda j: S.loc[i, j] * exp(-2 * H[i]) * I(y_l[j], 1)))
#     right = 0.5 * C * sum(j_u.map(lambda j: S.loc[i, j] * exp(H[j] - H[i])))
#
#     return left + right


def q_i(i, y_l, j_u, S, H, C):
    # i = x_i.index
    j_l = y_l.index
    # j_u = y_u.index

    S_i = S[i, j_l.values]
    I_j = y_l.map(partial(I, b=-1))
    left = sum(S_i * I_j * exp(2 * H[i]))

    S_i = S[i, j_u.values]
    exp_H_diff = (H[i] - H[j_u.values]).apply(exp)
    right = 0.5 * C * sum(S_i * exp_H_diff)

    return left + right


# def q_i(i, y_l, j_u, S, H, C):
#     # i = x_i.index
#     j_l = y_l.index
#     # j_u = y_u.index
#
#     left = sum(j_l.map(lambda j: S.loc[i, j] * exp(2 * H[i]) * I(y_l[j], -1)))
#     right = 0.5 * C * sum(j_u.map(lambda j: S.loc[i, j] * exp(H[i] - H[j])))
#
#     return left + right


def alpha(i_u, p, q, h):
    # i_u = y_u.index
    # p[i] = p_i(i, ...)
    numerator = sum(i_u.map(lambda i: p[i] * I(h[i], 1))) + sum(i_u.map(lambda i: q[i] * I(h[i], -1)))
    denominator = sum(i_u.map(lambda i: p[i] * I(h[i], -1))) + sum(i_u.map(lambda i: q[i] * I(h[i], 1)))

    return 0.25 * log(numerator / denominator)
