# -*- coding: utf-8 -*-
r""" Archimedean Clayton copula module

This module has a set of functions related with the bidimensional 
archimedean Clayton copula.

Functions:
----------
rvs : generates a random sampling (bidimensional) from a Clayton copula
fit : estimates the Clayton copula parameter from the data
pdf : calculates the pdf value for two realizations and a parameter of dep
cdf : calculates the cdf value for two realizations and a parameter of dep
aic : compute the aic value for a given Clayton cop and a data set
bic : compute the bic value for a given Clayton cop and a data set
upd : compute the value of the upper tail dependence of a Clayton copula
ltd : compute the value of the lower tail dependence of a Clayton copula

Notes:
----------
In this module, the Clayton copula is restricted uniquely to a 
bidimensional case (2d), which follows the nature of the Archimedean
copulas. 

Range of the dependence parameter:
$\theta \in \left[-1,\infty\right)\backslash\left\{0\right\}$

Generator function:
$\varphi\left(t\right) = \frac{1}{\theta}\left(t^{-\theta}-1\right)$

Copula function:
$C\left(u,v|\theta\right) = \left[\left(u^{-\theta}-1\right) + 
\left(v^{-\theta}-1\right)+1\right]^{-1/\theta}$

Created by: 
-----------


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


def rvs(theta, n):
    r"""Random sampling from the Clayton copula.

    This function draws random samples that follow the dependence 
    structure of a Clayton copula. It is restricted to a bivariate 
    dimension uniquely. 

    Parameters:
    ---------- 
    theta : float
        copula dependence parameter in [-1,inf)\{0}
    n : int
        numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follows a Clayton cop
    v : numpy.ndarray   
        vector of n elements on [0,1], which with u follows a Clayton cop
    Notes:
    ---------- 
    """
    
    u = st.uniform.rvs(loc=0, scale=1, size=n)
    q = st.uniform.rvs(loc=0, scale=1, size=n)

    v = (u**(-theta)*(q**(-theta/(1+theta))-1)+1)**(-1/theta)

    return u, v


def fit(x, y):
    r"""Clayton copula parameters from data.

    This function computes the copula parameter "theta" using the method
    of moments (relation kendall - copula parameter) from a given data
    set. 

    Parameters:
    ---------- 
    x : numpy.ndarray 
        First array of n elements
    y : numpy.ndarry
        Second array of n element
    Returns:
    ----------
    theta : float
        Copula parameters of dependence "theta" obtained from the data set
    Notes:
    ---------- 
    The relation is given by: 
    $\tau = \frac{\theta}{2+\theta}$
    """

    tau = st.kendalltau(x, y)[0]
    theta = 2*tau/(1 - tau)

    return theta


def cdf(u, v, theta):
    r"""CDF value of the Clayton Copula

    This function uses a Clayton copula with a given parameter of
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
        to a calculation of the Clayton copula cdf for the respective i
        element in u and v.
    Notes:
    ----------
    The CDF formula was taken from Zhang and Singh (2009). I haven't 
    reviewed the formula by myself yet.
    """

    cdf = np.max((u**(-theta) + v**(-theta) - 1)**(-1/theta), 0)

    return cdf


def pdf(u, v, theta):
    r"""PDF value of the Clayton copula

    This function uses a Clayton copula with a given parameter of
    dependence "theta" to compute the PDF of a pair or realizations
    u and v.

    Parameters:
    ---------- 
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta: float
        Parameter of dependence of the Clayton copula
    Returns:
    ----------
    pdf : numpy.ndarray
        Vector of n elements on [0,1], where each element i corresponds
        to a calculation of the Clayton copula pdf for the respective i
        element in u and v.
    Notes:
    ----------
    The PDF formula was taken from Zhang and Singh (2009). I haven't
    reviewed the formula by myself yet. The same formula appears in
    wikipedia.
    """

    pdf = np.zeros(u.shape)

    for i in range(len(pdf)):
        cdff = cdf(u[i], v[i], theta)
        if cdff > 0:
            num = (1 + theta)*(u[i]*v[i])**(-(1 + theta))
            den = (-1 + u[i]**(-theta) + v[i]**(-theta))**((1 + 2*theta)/theta)
            pdf[i] = num/den
        else:
            pdf[i] = 1

    return pdf


def aic(u, v, theta):
    r"""Clayton Copula AIC value

    This function uses a Clayton copula, with a given parameter of
    dependence "theta", to compute the Akaike Information Criterion 
    (AIC) from a given dataset

    Parameters:
    ---------- 
    u : numpy.ndarray 
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Clayton copula 
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
    r"""Clayton Copula BIC value

    This function uses a Clayton copula, with a given parameter of
    dependence "theta", to compute the Bayesian Information Criterion 
    (BIC) from a given dataset

    Parameters:
    ---------- 
    u : numpy.ndarray 
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Clayton copula
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
    r"""Clayton Copula value of upper tail dependence

    This function computes the upper tail dependence value of a Clayton
    copula with a parameter of dependence "theta".

    Parameters:
    ---------- 
    theta : float
        Parameter of dependence of the Clayton copula
    Returns:
    ----------
    utd : float
        Value of upper tail dependence for the given Clayton copula
    Notes:
    ----------
    Clayton copulas, independently of the parameter of dependence theta, do
    not have upper tail dependence.
    """

    utd = 0

    return utd


def ltd(theta):
    r"""Clayton Copula value of lower tail dependence

    This function computes the lower tail dependence value of a Clayton
    copula with a parameter of dependence "theta".

    Parameters:
    ----------
    theta : float
        Parameter of dependence of the Clayton copula
    Returns:
    ----------
    ltd : float
        Value of lower tail dependence for the given Clayton copula
    Notes:
    ----------
    """

    ltd = 2**(-1/theta)

    return ltd

