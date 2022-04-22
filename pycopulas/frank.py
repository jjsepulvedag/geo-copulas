# -*- coding: utf-8 -*-
r""" Archimedean Frank copula module

This module has a set of functions related with the bidimensional 
archimedean Frank copula.

Functions:
----------
rvs : generates a random sampling (bidimensional) from a Frank copula
cdf : calculates the cdf value for two realizations and a parameter of dep
pdf : calculates the pdf value for two realizations and a parameter of dep
fit : estimates the Frank copula parameter from the data
aic : compute the aic value for a given Frank cop and a data set
bic : compute the bic value for a given Frank cop and a data set
upd : compute the value of the upper tail dependence of a Frank copula
ltd : compute the value of the lower tail dependence of a Frank copula

Notes:
----------
In this module, the Frank copula is restricted uniquely to a 
bidimensional case (2d), which follows the nature of the Archimedean
copulas. Note that when the dependence parameter is equal to zero, 
we obtain the independence copula. Additionally, Frank copula is a 
comprehensive copula, that is to say lim theta -> -inf = countermonotonic
copula, and lim theta -> -inf = comonotonic copula. 

Range of the dependence parameter:
$\theta \in \left(-\infty,+\infty\right)\backslash\left\{0\right\}$

Generator function:
$\varphi\left(t\right) = 
-\ln\left[\frac{e^{-t\theta}-1}{e^{-\theta}-1}\right]$

Copula function:
$C\left(u,v,\theta\right) = 
-\frac{1}{\theta}\ln\left[1+\frac{\left(e^{-u\theta}-1\right)
\left(e^{-v\theta}-1\right)}{e^{-\theta}-1}\right]$

References:
----------
1."An introduction to copulas"
Nelsen Robert
2006
Springer science and business media
2."Risk and reliability in geotechnical engineering"
Phoon Kok-Kwang, Ching Jianye
2014
CRC Press
3. "Copulas and Their Applications in Water Resources Engineering"
Zhang Lan, Singh Vijay 
2019
Cambridge University Press
"""

import numpy as np
import scipy.stats as st
import scipy.optimize as sop
import scipy.integrate as sint


def rvs(theta, n):
    r"""Random sampling from the Frank Copula.

    This function draws random samples that follow the dependence
    structure of a Frank copula.

    Parameters:
    ----------
    theta : float
        Copula dependence parameter in (-inf,inf)
    n : int
        Numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follows a Frank cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follows a Frank cop
    Notes:
    ----------
    """

    u = st.uniform.rvs(loc=0, scale=1, size=n)
    w = st.uniform.rvs(loc=0, scale=1, size=n)

    v = (-(1/theta) * np.log(1 + w*(1-np.exp(-theta))
         / (w*(np.exp(-theta*u)-1)-np.exp(-theta*u))))

    return u, v


def cdf(u, v, theta):
    r"""CDF value of the Frank Copula

    This function uses a Frank copula with a given parameter of
    dependence "theta" to compute the CDF of a pair or realizations
    u and v.

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta: float
        Copula parameters of dependence
    Returns:
    ----------
    cdf : numpy.ndarray
        Vector of n elements on [0,1], where each element i corresponds
        to a calculation of the Frank copula cdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    num = (np.exp(-theta*u)-1)*(np.exp(-v*theta)-1)
    den = (np.exp(-theta)-1)

    cdf = -(1/theta)*np.log(1 + num/den)

    return cdf


def pdf(u, v, theta):
    r"""PDF value of the Frank copula

    This function uses a Frank copula with a given parameter of
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
        to a calculation of the Frank copula pdf for the respective i
        element in u and v.
    Notes:
    ----------
    The PDF formula was taken from Phoon and Ching (2014). I haven't
    reviewed the formula by myself yet.
    """

    num = -theta*(np.exp(-theta) - 1)*np.exp(-theta*(u + v))
    den = ((np.exp(-theta)-1)+(np.exp(-theta*u)-1)*(np.exp(-theta*v)-1))**2

    pdf = num/den

    return pdf


def fit(vals=(None, None), corr=None, dists=(None, None), method='moments-t'):
    r"""Frank Copula parameters adjustment.

    This function adjusts the Frank copula (and marginals if required) from
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
        This list contains the values of adjustment to the Frank copula. If
        the moments-t or CML method are used, this list will contains only one
        element corresponding to the copula parameter of dependence. If the
        MLE or IFM method are selected, this list contains, in order, the
        copula copula parameter of dependence, the values of adjustment of the
        first univariate distribution, and the values of adjustment of the
        second univariate distribution.
    Notes:
    ----------
    The tau-theta relation is given by the equation given in [3], where the
    Debye function must be used.
    """

    if method.lower() == 'moments-t':

        def bisection(tau):
            def function(theta):
                def debye(t):
                    value = t/(np.exp(t) - 1)
                    return value
                integration = sint.quad(debye, a=0, b=theta)[0]/theta
                return tau - (1 - (4/theta)*(1-integration))

            if -1+1e-3 < tau < 1-1e-3:
                sol = sop.bisect(function, -500, 501)
                return sol
            elif tau <= -1+1e-3:
                return -np.inf
            elif 1-1e-3 <= tau:
                return np.inf

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

        theta = bisection(tau)
        return theta

    elif method.upper() == 'MLE':
        return None

    elif method.upper() == 'IFM':
        return None

    elif method.upper() == 'CML':
        return None


def aic(u, v, theta):
    r"""Frank Copula AIC value

    This function uses a Frank copula, with a given parameter of
    dependence "theta", to compute the Akaike Information Criterion
    (AIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Frank copula
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
    r"""Frank Copula BIC value

    This function uses a Frank copula, with a given parameter of
    dependence "theta", to compute the Bayesian Information Criterion
    (BIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Frank copula
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
    """Value of the upper tail dependence of the Frank copula"""

    return 0


def ltd(theta):
    """Value of the lower tail dependence of the Frank copula"""

    return 0
