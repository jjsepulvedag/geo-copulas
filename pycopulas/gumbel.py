# -*- coding: utf-8 -*-
r""" Archimedean Gumbel copula module.

This module has a set of functions related with the bidimensional
archimedean Gumbel copula.

Functions:
----------
rvs : generates a random sampling (bidimensional) from a Gumbel copula
cdf : calculates the cdf value for two realizations and a parameter of dep
pdf : calculates the pdf value for two realizations and a parameter of dep
fit : estimates the Gumbel copula parameter from the data
aic : compute the aic value for a given Gumbel cop and a data set
bic : compute the bic value for a given Gumbel cop and a data set
upd : compute the value of the upper tail dependence of a Gumbel copula
ltd : compute the value of the lower tail dependence of a Gumbel copula

Notes:
----------
In this module, the Gumbel copula is restricted uniquely to a
bidimensional case (2d), which follows the nature of the Archimedean
copulas. When theta tends to 1, we obtain the independence copula, and
when theta tends to infinity, we obtain the comonotonicity copula

Range of the dependence parameter:
$\theta \in \left[1,\infty\right)$

Generator function:
$\varphi\left(t\right) = \left(-\ln\left(t\right)\right)^\theta$

Copula function:
$C\left(u,v,\theta\right) = \exp\left\{-\left[\left(-\ln u\right)^{\theta}+
\left(-\ln v\right)^{\theta}\right]^{1/\theta}\right\}$

References:
----------
1."An introduction to copulas"
Nelsen Robert
2006
Springer science and business media
2. "Copulas and Their Applications in Water Resources Engineering"
Zhang Lan and Singh Vijay P
2009
Cambridge University Press
"""

import numpy as np
import scipy.stats as st
import scipy.optimize as sop


def rvs(theta, n):
    r"""Random sampling from the Gumbel copula

    This function draws random samples that follow the dependence
    structure of a Gumbel copula. It is restricted to a bivariate
    dimension uniquely.

    Parameters:
    ----------
    theta : float
        copula dependence parameter in [1,inf)
    n : int
        numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follows a Gumbel cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follows a Gumbel cop
    Notes:
    ----------
    """

    def bisection(k):
        def function(t):
            return k - t*(1-np.log(t)/theta)
        solution = sop.bisect(function, 0.0000000001, 0.9999999)
        return solution

    s = st.uniform.rvs(loc=0, scale=1, size=n)
    q = st.uniform.rvs(loc=0, scale=1, size=n)

    t = np.fromiter(map(bisection, q), dtype=np.float64)

    generator = (-np.log(t))**theta

    u = np.exp(-(s*generator)**(1/theta))
    v = np.exp(-((1-s)*generator)**(1/theta))

    return u, v


def cdf(u, v, theta):
    r"""CDF value of the Gumbel Copula

    This function uses a Gumbel copula with a given parameter of
    dependence "theta" to compute the CDF of a pair or realizations
    u and v

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta: float
        Copula parameter of dependence
    Returns:
    ----------
    cdf : numpy.ndarray
        Vector of n elements on [0,1], where each element i corresponds
        to a calculation of the Gumbel copula cdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    cdf = np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

    return cdf


def pdf(u, v, theta):
    r"""PDF value of the Gumbel copula

    This function uses a Gumbel copula with a given parameter of
    dependence "theta" to compute the PDF of a pair or realizations
    u and v.

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    Returns:
    ----------
    pdf : numpy.ndarray
        Vector of n elements on [0,1], where each element i corresponds
        to a calculation of the Gumbel copula pdf for the respective i
        element in u and v.
    Notes:
    ----------
    The PDF formula was taken from Zhang and Singh (2009). I haven't
    reviewed the formula by myself yet. This copula differs from the one
    that appears in wikipedia.
    """

    w = (-np.log(u))**theta + (-np.log(v))**theta

    num = ((-np.log(v))**(-1 + theta)*(-np.log(u))**(-1 + theta) *
            (w**((2 - 2*theta)/theta) - (1 - theta)*w**((1 - 2*theta)/theta)))
    den = u*v*np.exp(w**(1/theta))

    pdf = num/den

    return pdf


def fit(vals=(None, None), corr=None, dists=(None, None), method='moments-t'):
    r"""Gumbel Copula parameters adjustment.

    This function adjusts the Gumbel copula (and marginals if required) from
    a given dataset or from a given Kendall's tau coefficient.

    Parameters:
    ----------
    vals : tuple
        Tuple of two elements, where each one is a numpy.ndarray of n elements.
        This tuple is optional for the moments-t method, but mandatory for the
        MLE, IFM and CML methods.
    corr : float
        Float value within [-1, 1]. It corresponds to a Kendall's tau
        coefficient. This variable is optional for the moments-t method, and
        it is not applicable to the MLE, IFM and CML methods.
    dists : tuple
        Tuple of two elements, where each element is a string that corresponds
        to a probability distribution. Options are: 'normal', 'truncnormal',
        'lognormal', 'gamma', 'beta', 'weibull', 'gumbel'. This variable is
        only applicable to MLE and IFM methods.
    method : string
        There are 4 method in total, namely (1) 'moments-t' that computes the
        copula parameter of dependence by means of the Kendall's tau
        coefficient, and requires the definition of the variable vals OR corr,
        (2) 'MLE' and (3) 'IFM' methods compute both copula parameter of
        dependence and marginals parameters, by the maximum likelihood
        methodology, so these require the mandatory definition of vals and
        dists variables, and finally, (4) CML method computes the copula
        parameter of dependence, by means of the maximum likelihood, and only
        requires the definition of vals variable.
    Returns:
    ----------
    theta : list
        This list contains the values of adjustment to the Gumbel copula. If
        the moments-t or CML method are used, this list will contains only one
        element corresponding to the copula parameter of dependence. If the
        MLE or IFM method are selected, this list contains, in order, the
        copula copula parameter of dependence, the values of adjustment of the
        first univariate distribution, and the values of adjustment of the
        second univariate distribution.
    Notes:
    ----------
    """

    if method.lower() == 'moments-t':

        if all(isinstance(i, (np.ndarray, list)) for i in vals):
            tau = np.round(st.kendalltau(vals[0], vals[1])[0], 3)
        elif isinstance(corr, (float, int)) and (-1 <= corr <= 1):
            tau = np.round(corr, 3)
        else:
            message = "Invalid values encountered in 'vals' or 'corr' " \
             "variables. Remember that 'vals' must be a tuple with two " \
             "elements where each element is a numpy.ndarray, and 'corr '" \
             "be a float number within [-1, 1]"

            raise ValueError(message)

        theta = 1/(1 - tau)
        return theta

    elif method.upper() == 'MLE':
        return None

    elif method.upper() == 'IFM':
        return None

    elif method.upper() == 'CML':
        return None


def aic(u, v, theta):
    r"""Gumbel Copula AIC value

    This function uses a Gumbel copula, with a given parameter of
    dependence "theta", to compute the Akaike Information Criterion
    (AIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Gumbel copula
    Returns:
    ----------
    aic : float
        AIC value obtained from the copula and certain dataset
    Notes:
    ----------
    """

    k = 1  # number of estimated parameters = 1, i.e. theta

    aic = 2*k - 2*np.sum(np.log(pdf(u, v, theta)))

    return aic


def bic(u, v, theta):
    r"""Gumbel Copula BIC value

    This function uses a Gumbel copula, with a given parameter of
    dependence "theta", to compute the Bayesian Information Criterion
    (BIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Gumbel copula
    Returns:
    ----------
    bic : float
        BIC value obtained from the copula and certain dataset
    Notes:
    ----------
    """

    k = 1 # number of estimated parameters = 1, i.e. theta
    bic = k*np.log(len(u)) - 2*np.sum(np.log(pdf(u, v, theta)))

    return bic


def utd(theta):
    r"""Gumbel Copula value of upper tail dependence

    This function computes the upper tail dependence value of a Gumbel
    copula with a parameter of dependence "theta".

    Parameters:
    ----------
    theta : float
        Parameter of dependence of the Gumbel copula
    Returns:
    ----------
    utd : float
        Value of upper tail dependence for the given Gumbel copula
    Notes:
    ----------
    """

    utd = 2 - 2**(1/theta)

    return utd


def ltd(theta):
    r"""Gumbel Copula value of lower tail dependence

    This function computes the lower tail dependence value of a Gumbel
    copula with a parameter of dependence "theta".

    Parameters:
    ----------
    theta : float
        Parameter of dependence of the Gumbel copula
    Returns:
    ----------
    ltd : float
        Value of lower tail dependence for the given Gumbel copula
    Notes:
    ----------
    Gumbel copulas, independently of the parameter of dependence theta, do
    not have lower tail dependence.
    """

    ltd = 0

    return ltd
