"""
Exact Berk-Jones statistics for goodness-of-fit testing.
This code includes computation of the M_n, M_n^+ and M_n^- statistics.

See paper for details: http://arxiv.org/abs/1311.3190

To use this code, you must install the Python language interpreter and then install the NumPy and SciPy packages.
Alternatively, you can download Anaconda by Continuum Analytics. It is a free python installer that includes NumPy, SciPy and other packages.

Written by Amit Moscovich, Weizmann Institute of Science, 12/2013.
Send any questions or comments to: moscovich (at) gmail (dot) com
"""
from numpy import arange, sort, asarray
from scipy.special import betainc

def _uniform_order_statistic_cdf(i, n, t):
    """
    _uniform_order_statistic_cdf(i, n, t) -> Pr[U_(i+1) < t]
    
    Let U_1, ..., U_n ~ Uniform[0,1] be n independent random variables
    and let U_(1) < ... < U_(n) denote the same variables in sorted order.
    This function returns the Cumulative Distribution function of U_(i+1),
    i.e. Pr[U_(i+1) < t]

    note that this function also works for numpy array inputs"""
    return betainc(i+1, n-i, t)

def _compute_p_values(uniform01_samples):
    assert all(0 <= x <= 1 for x in uniform01_samples)
    n = len(uniform01_samples)
    sorted_uniform01_samples = sort(uniform01_samples)
    p_values = _uniform_order_statistic_cdf(arange(n), n, sorted_uniform01_samples)
    return p_values

def Mn(uniform01_samples):
    """Mn(uniform01_samples) -> CKS statistic (two sided)

    Compute the two sided Mn test statistic for a list of samples assumed to come from a known (and fully specified) continuous distribution F.

    Args:
        uniform01_samples: data samples after being transformed using the cumulative distribution of the null hypothesis. This should be a list of numbers in the range 0 to 1.

    Returns:
        The two-sided M_n statistic defined in Eq. (3.2) of http://arxiv.org/pdf/1311.3190v5.pdf

    Usage example:
        To test goodness of fit to a fully specified cumulative distribution function F,
        apply the distribution function to your samples and then use cks().
        e.g. to compute the CKS goodness of fit to a standard normal distribution, use:
            >>> from numpy import array
            >>> from scipy.stats import norm
            >>> cks(norm.cdf(numpy.array([1,2,3])))
    """
    p_values = _compute_p_values(uniform01_samples) 
    return min(p_values.min(), 1.0-p_values.max())

def Mn_plus(uniform01_samples):
    """Mn_plus(uniform01_samples) -> Mn+ score (one-sided sided)

    Like the Mn() function, but for testing deviations towards lower values only.
    """
    p_values = _compute_p_values(uniform01_samples) 
    return p_values.min()

def Mn_minus(uniform01_samples):
    """Mn_minus(uniform01_samples) -> Mn- score (one-sided sided)

    Like the Mn() function, but for testing deviations towards higher values only.
    """
    p_values = _compute_p_values(uniform01_samples) 
    return 1-p_values.max()
