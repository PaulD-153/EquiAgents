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

def variance_penalty_gradient(expected_claims):
    n = len(expected_claims)
    mean = sum(expected_claims) / n
    gradient = [2 * (c - mean) / n for c in expected_claims]
    return gradient

def variance_penalty_numpy(values):
    values = np.array(values)
    mean = np.mean(values)
    return np.mean((values - mean) ** 2)
