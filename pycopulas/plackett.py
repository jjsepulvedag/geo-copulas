# -*- coding: utf-8 -*-
r"""Plackett copula module

This module has a set of functions related with the bidimensional
Plackett copula.

Functions:
----------
rvs : generates a random sampling (bidimensional) from a Plackett copula
pdf : calculates the pdf value for two realizations and a parameter of dep
cdf : calculates the cdf value for two realizations and a parameter of dep
fit : estimates the Plackett copula parameter from the data
aic : compute the aic value for a given Plackett cop and a data set
bic : compute the bic value for a given Plackett cop and a data set
upd : compute the value of the upper tail dependence of a Plackett copula
ltd : compute the value of the lower tail dependence of a Plackett copula

Notes:
----------
In this module, the Plackett copula is restricted uniquely to a
bidimensional case (2d). When the copula parameter "theta" is equal to 1,
we obtain the independence copula

Range of the dependence parameter:
$\theta \in \left(0,\infty\right)\backslash\left\{1\right\}$

Copula function:
$C\left(u,v|\theta\right) = \frac{1 + \left(\theta-1\right)\left(u+v\right)-
\sqrt{\left[1 + \left(\theta-1\right)\left(u+v\right)\right]^2 -
4uv\theta\left(\theta-1\right)}}{2\left(\theta-1\right)}$

References:
----------
1. "An introduction to copulas"
Nelsen Roger B
2006
Springer Science and Business Media
2."Risk and reliability in geotechnical engineering"
Phoon Kok-Kwang and Ching Jianye
2014
CRC Press
3. "Copulas and Their Applications in Water Resources Engineering"
Zhang Lan and Singh Vijay P
2009
Cambridge University Press
"""

import numpy as np
import scipy.stats as st
import scipy.optimize as sop
from scipy.integrate import dblquad


def rvs(theta, n):
    """Random sampling from Plackett copula.

    This function draws random samples that follow the dependence
    structure of a Plackett copula. It is restricted to a bivariate
    dimension uniquely.

    Parameters:
    ----------
    theta : float
        Parameter of dependence of the bivariate Plackett copula
    n : int
        Numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follow a Plackett cop
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follow a Plackett cop
    Notes:
    ----------
    """

    n = int(n)

    u = st.uniform.rvs(loc=0, scale=1, size=n)
    t = st.uniform.rvs(loc=0, scale=1, size=n)

    a = t*(1 - t)
    b = theta + a*(theta - 1)**2
    c = 2*a*(u*theta**2 + 1 - u) + theta*(1 - 2*a)
    d = np.sqrt(theta)*np.sqrt(theta + 4*a*u*(1 - u)*(1 - theta)**2)

    v = (c - (1 - 2*t)*d)/(2*b)

    return u, v


def cdf(u, v, theta):
    r"""CDF value of the Plackett Copula

    This function uses a Plackett copula with a given parameter of
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
        to a calculation of the Plackett copula cdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    s = 1 + (theta - 1)*(u + v)
    cdf = (s - np.sqrt(s**2 - 4*u*v*theta*(theta - 1)))/(2*(theta - 1))

    return cdf


def pdf(u, v, theta):
    r"""PDF value of the Plackett copula

    This function uses a Plackett copula with a given parameter of
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
        to a calculation of the Plackett copula pdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    num = theta*(1 + (theta - 1)*(u + v - 2*u*v))
    den = ((1 + (theta - 1)*(u + v))**2 - 4*u*v*theta*(theta - 1))**(3/2)

    pdf = num/den

    return pdf


def fit(vals=(None, None), corr=None, dists=(None, None), method='moments-s'):
    r"""Plackett Copula parameters adjustment.

    This function adjusts the Plackett copula (and marginals if required) from
    a given dataset or from a given Spearman's rho or Kendall's tau
    coefficient.

    Parameters:
    ----------
    vals : tuple
        Tuple of two elements, where each one is a numpy.ndarray of n elements.
        This tuple is optional for the moments-s and moments-t methods, but
        mandatory for the MLE, IFM and CML methods.
    corr : float
        Float value within [-1, 1]. It corresponds to a Spearman's Rho or
        Kendall's tau coefficient. This variable is optional for the moments-s
        and moments-t methods, and it is not applicable to the MLE, IFM and CML
        methods.
    dists : tuple
        Tuple of two elements, where each element is a string that corresponds
        to a probability distribution. Options are: 'normal', 'truncnormal',
        'lognormal', 'gamma', 'beta', 'weibull', 'gumbel'
    method : string
        There are 5 method in total, namely (1) 'moments-s' and (2) 'moments-t'
        methods which compute the copula parameter of dependence by means of
        the Spearman's rho or Kendall's tau coefficient respectively, and
        requires the definition of the variable vals OR corr, (3) 'MLE' and (4)
        'IFM' methods compute both copula parameter of dependence and marginals
        parameters, by the maximum likelihood methodology, so these require the
        mandatory definition of vals and dists variables, and finally, (5) CML
        method computes the copula parameter of dependence, by means of the
        maximum likelihood, and only requires the definition of vals
    Returns:
    ----------
    theta : list
        This list contains the values of adjustment to the Plackett copula. If
        the moments-s, moments-t or CML method are used, this list will
        contains only one element corresponding to the copula parameter of
        dependence. If the MLE or IFM method are selected, this list contains,
        in order, the copula copula parameter of dependence, the values of
        adjustment of the first univariate distribution, and the values
        of adjustment of the second univariate distribution.
    Notes:
    ----------
    Methods 'moments-s' and 'moments-t' are solved numerically. Specially,
    'moments-t' uses the generic function for all copulas where a Riemann
    Stieltjes  integral must be solved (see [1]).

    """

    # moments-s method --------------------------------------------------------
    if method.lower() == 'moments-s':

        def bisection(ps):

            def funct(theta, ps):
                # Function to solve numerically.
                f = (ps - (theta + 1)/(theta - 1) +
                     (2*theta*np.log(theta))/(theta - 1)**2)
                return f

            if -1 <= ps < 0:
                if -1 + 1e-8 < ps < 0:
                    solution = sop.bisect(funct, 0+1e-10, 1-1e-10, args=(ps))
                else:
                    solution = 0
            elif 0 < ps <= 1:
                if ps <= 1 - 1e-8:
                    solution = sop.bisect(funct, 1+1e-10, 1e10, args=(ps))
                else:
                    solution = np.inf
            else:
                solution = 1
            return solution

        if all(isinstance(i, (np.ndarray, list)) for i in vals):
            ps = np.round(st.spearmanr(vals[0], vals[1])[0], 3)
            theta = bisection(ps)
        elif isinstance(corr, (float, int)) and (-1 <= corr <= 1):
            ps = np.round(corr, 3)
            theta = bisection(ps)
        else:
            message = "Invalid values encountered in 'vals' or 'corr' " \
             "variables. Remember that 'vals' must be a tuple with two " \
             "elements where each element is a numpy.ndarray, and 'corr '" \
             "be a float number within [-1, 1]"

            raise ValueError(message)

        return theta

    # moments-t method --------------------------------------------------------
    elif method.lower() == 'moments-t':

        def integral(u, v, theta):
            # Riemann–Stieltjes integral, defined in [1], that relates
            # tau-theta.

            f1 = cdf(u, v, theta)
            f2 = pdf(u, v, theta)
            return f1*f2

        def funct(theta, tau):
            # Function to solve numerically.

            f = (tau - 4*dblquad(integral, 0, 1, lambda x: 0,
                                 lambda x: 1, args=[theta])[0] + 1)
            return f

        def bisection(tau):
            if tau == 0:
                solution = 1
            elif -1 <= tau < 0:
                if tau > -0.98:  # Set 0.98 to avoid IntegrationWarning errors
                    solution = sop.bisect(funct, 0.0001, 0.9999, args=(tau))
                else:
                    solution = 0
            elif 0 < tau <= 1:
                if tau < 0.98:
                    solution = sop.bisect(funct, 1.0001, 10000, args=(tau))
                else:
                    solution = np.inf

            return solution

        if all(isinstance(i, (np.ndarray, list)) for i in vals):
            tau = np.round(st.kendalltau(vals[0], vals[1])[0], 3)
            theta = bisection(tau)
        elif isinstance(corr, (float, int)) and (-1 <= corr <= 1):
            tau = np.round(corr, 3)
            theta = bisection(tau)
        else:
            message = "Invalid values encountered in 'vals' or 'corr' " \
             "variables. Remember that 'vals' must be a tuple with two " \
             "elements where each element is a numpy.ndarray, and 'corr '" \
             "be a float number within [-1, 1]"
            raise ValueError(message)

        return theta

    elif method.upper() == 'MLE':
        theta = None

    elif method.upper() == 'IFM':
        theta = None

    elif method.upper() == 'CML':
        theta = None

    return theta


def aic(u, v, theta):
    r"""Plackett Copula AIC value

    This function uses a Plackett copula, with a given parameter of
    dependence "theta", to compute the Akaike Information Criterion
    (AIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Plackett copula
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
    r"""Plackett Copula BIC value

    This function uses a Plackett copula, with a given parameter of
    dependence "theta", to compute the Bayesian Information Criterion
    (BIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Plackett copula
    Returns:
    ----------
    bic : float
        BIC value obtained from the copula and certain dataset
    Notes:
    ----------
    """

    k = 1  # number of estimated parameters = 1, i.e. theta

    bic = k*np.log(len(u)) - 2*np.sum(np.log(pdf(u, v, theta)))

    return bic


def utd(theta):
    """Value of the upper tail dependence of the Plackett copula"""

    return 0


def ltd(theta):
    """Value of the lower tail dependence of the Plackett copula"""

    return 0
