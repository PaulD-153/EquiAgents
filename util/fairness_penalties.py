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

def jain_index_penalty(claims):
    """
    Jain surrogate using variance.
    Note: Jain true formula is non-convex. We use variance as convex surrogate.
    """
    return variance_penalty(claims)

def gini_penalty(claims):
    """
    Gini surrogate using mean absolute deviation (MAD).
    """
    n = len(claims)
    x = cv.vstack(claims)
    mean = cv.sum(x) / n
    mad = cv.sum(cv.abs(x - mean))
    return mad

def nsw_penalty(claims):
    """
    Nash Social Welfare convex surrogate via log-sum.
    """
    x = cv.vstack(claims)
    return -cv.sum(cv.log(x + 1e-6))  # divide by n not necessary in optimization

def minshare_penalty(claims):
    """
    Min-share fairness: penalize small minimum allocation.
    """
    x = cv.vstack(claims)
    return -cv.min(x)

def envy_scaled_penalty(claims, reward_scaling):
    """
    Scaled envy surrogate. We compute a convex approximation by summing scaled absolute deviations from mean.
    """
    n = len(claims)
    x = cv.vstack([claims[i] * reward_scaling[i] for i in range(n)])  # scaled vector
    mean = cv.sum(x) / n
    return cv.sum(cv.abs(x - mean))
