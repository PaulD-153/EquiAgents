import cvxpy as cv
import numpy as np

def variance_penalty(claims):
    """
    Convex surrogate: variance of claims.
    """
    n = len(claims)
    x = cv.vstack(claims)
    mean = cv.sum(x) / n
    diffs = x - mean
    return cv.sum_squares(diffs)

def variance_penalty_gradient(x):
    """
    Gradient of variance with respect to x.
    Input: x = list or array of agent claims
    Output: gradient vector
    """
    x = np.array(x)
    mean = np.mean(x)
    n = len(x)
    return (2.0 / n) * (x - mean)

def variance_penalty_numpy(values):
    values = np.array(values)
    mean = np.mean(values)
    return np.mean((values - mean) ** 2)
