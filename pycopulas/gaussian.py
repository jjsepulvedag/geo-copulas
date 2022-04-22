# -*- coding: utf-8 -*-
r"""Gaussian copula module

This module has a set of functions related with the bidimensional
Gaussian copula.

Functions:
----------
rvs : generates a random sampling (bidimensional) from a Gaussian copula
pdf : calculates the pdf value for two realizations and a parameter of dep
cdf : calculates the cdf value for two realizations and a parameter of dep
fit : estimates the Gaussian copula parameter from the data
aic : compute the aic value for a given Gaussian cop and a data set
bic : compute the bic value for a given Gaussian cop and a data set
upd : compute the value of the upper tail dependence of a Gaussian copula
ltd : compute the value of the lower tail dependence of a Gaussian copula

Notes:
----------
In this module, the Gaussian copula is restricted uniquely to a
bidimensional case (2d). Future work will have to include an extension to
more than two dimensions. When the copula parameter is equal to the
identity matrix, we obtain the independence copula.

Range of the dependence parameter:
$\theta \in \left[-1,1\right]$

Copula function:
$C\left(u,v|\theta\right) =
\Phi_\theta\left(\Phi^{-1}\left(u\right),\Phi^{-1}\left(v\right)\right)$

References:
----------
1."Modelling dependence with Copulas, and applications to risk management"
Embrechts Paul et al.
2001
Department of mathematics, ETHZ
2."Risk and reliability in geotechnical engineering"
Phoon Kok-Kwang and Ching Jianye
2014
CRC Press
"""

import numpy as np
import scipy.stats as st
import scipy.optimize as optimize


def rvs(p, n):
    """Random sampling from Gaussian copula.

    This function draws random samples that follow the dependence
    structure of a Gaussian copula. It is restricted to a bivariate
    dimension uniquely.

    Parameters:
    ----------
    p : float
        pearson's rho correlation coefficient
    n : int
        numbers of samples to be generated
    Returns:
    ----------
    u : numpy.ndarray
        vector of n elements on [0,1], which with v follow a Gaussian cop.
    v : numpy.ndarray
        vector of n elements on [0,1], which with u follow a Gaussian cop.
    Notes:
    ----------
    """

    R = np.array([[1, p], [p, 1]])
    A = np.linalg.cholesky(R)
    x = np.zeros((n, 2))

    for i in range(n):
        z = st.norm.rvs(loc=0, scale=1, size=2)
        x[i] = np.matmul(A, z)

    u = st.norm.cdf(x)
    return u[:, 0], u[:, 1]


def cdf(u, v, theta):
    r"""CDF value of the Gaussian Copula

    This function uses a Gaussian copula with a given parameter of
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
        to a calculation of the gaussian copula cdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    c = np.array([[1, theta], [theta, 1]])
    m = np.array([0, 0])

    u_inv = st.norm.ppf(u, loc=0, scale=1)
    v_inv = st.norm.ppf(v, loc=0, scale=1)
    array = np.array([u_inv, v_inv])

    cdf = st.multivariate_normal.cdf(array, mean=m, cov=c)

    return cdf


def pdf(u, v, theta):
    r"""PDF value of the Gaussian copula

    This function uses a Gaussian copula with a given parameter of
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
        to a calculation of the Gaussian copula pdf for the respective i
        element in u and v.
    Notes:
    ----------
    """

    u_inv = st.norm.ppf(u, loc=0, scale=1)
    v_inv = st.norm.ppf(v, loc=0, scale=1)

    f1 = (1/np.sqrt(1 - theta**2))
    f2 = (np.exp(- ((u_inv**2)*theta**2 - 2*u_inv*v_inv*theta +
          (v_inv**2)*theta**2)/(2*(1 - (theta**2)))))

    pdf = f1*f2

    return pdf


def fit(vals=(None, None), corr=None, dists=(None, None), method='moments-t'):
    r"""Gaussian Copula parameters adjustment.

    This function adjusts the Gaussian copula (and marginals if required) from
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
        dists variable, and finally, (4) CML method computes the copula
        parameter of dependence, by means of the maximum likelihood, and only
        requires the definition of vals.
    Returns:
    ----------
    theta : list
        This list contains the values of adjustment to the Gaussian copula. If
        the moments-t or CML method are used, this list will contains only one
        element corresponding to the copula parameter of dependence. If the
        MLE or IFM method are selected, this list contains, in order, the
        copula copula parameter of dependence, the values of adjustment of the
        first univariate distribution, and the values of adjustment of the
        second univariate distribution.
    Notes:
    ----------
    The tau-theta relation is given by:
    $\tau = \frac{2}{\pi}\sin^{-1}\left(\rho\right)$
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

        theta = np.sin(tau*np.pi/2)
        return theta

    elif method.upper() == 'MLE':
        return None

    elif method.upper() == 'IFM':
        # results = {marginals[0]: None, marginals[1]: None,
        #            'Gaussian copula': None, 'Log_Likelihood': None}
        # data = [x, y]
        # dist = []
        # for i in range(len(marginals)):
        #     if marginals[i].lower() == 'normal':
        #         loc, scale = st.norm.fit(data[i])
        #         results[marginals[i]] = (loc, scale)
        #         dist.append(st.norm(loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'lognormal':
        #         s, loc, scale = st.lognorm.fit(data[i], floc=0)
        #         results[marginals[i]] = (s, loc, scale)
        #         dist.append(st.lognorm(s=s, loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'exponential':
        #         loc, scale = st.expon.fit(data[i], floc=0)
        #         results[marginals[i]] = (loc, scale)
        #         dist.append(st.expon(loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'gamma':
        #         a, loc, scale = st.gamma.fit(data[i], floc=0)
        #         results[marginals[i]] = (a, loc, scale)
        #         dist.append(st.gamma(a=a, loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'beta':
        #         min_val = 0
        #         max_val = np.max(data[i]) + 10
        #         a, b, loc, scale = st.beta.fit(data[i], floc=min_val,
        #                                        fscale=max_val - min_val)
        #         results[marginals[i]] = (a, b, loc, scale)
        #         dist.append(st.beta(a=a, b=b, loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'weibull':
        #         c, loc, scale = st.weibull_min.fit(data[i], floc=0)
        #         results[marginals[i]] = (c, loc, scale)
        #         dist.append(st.weibull_min(c=c, loc=loc, scale=scale))
        #         continue
        #     elif marginals[i].lower() == 'gumbel':
        #         loc, scale = st.gumbel_r.fit(data[i])
        #         results[marginals[i]] = (loc, scale)
        #         dist.append(st.gumbel_r(loc=loc, scale=scale))
        #         continue

        # u = dist[0].cdf(x)
        # v = dist[1].cdf(y)

        # def loglik_copula(theta, u, v):
        #     L = - np.sum(np.log(pdf(u, v, theta)))
        #     return L
        # theta = optimize.minimize(loglik_copula, 0, args=(u, v),
        #                           bounds=[(-0.99, 0.99)]).x
        # results['Gaussian copula'] = theta
        # results['Log_Likelihood'] = np.sum(np.log(pdf(u, v, theta)))

        # return results
        return None

    elif method.upper() == 'CML':
        # results = {'Gaussian_copula': None, 'log_likelihood': None}
        # u = st.rankdata(x)/(len(x) + 1)
        # v = st.rankdata(y)/(len(y) + 1)

        # def cml(theta, u, v):
        #     L = - np.sum(np.log(pdf(u, v, theta)))
        #     return L
        # theta = optimize.minimize(cml, 0, args=(u, v), bounds=[(-0.99, 0.99)])
        # results['Gaussian_copula'] = theta.x
        # results['log_likelihood'] = np.sum(np.log(pdf(u, v, theta.x)))
        # return results
        return None


def aic(u, v, theta):
    r"""Gaussian Copula AIC value

    This function uses a Gaussian copula, with a given parameter of
    dependence "theta", to compute the Akaike Information Criterion
    (AIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Gaussian copula (Pearson's rho)
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
    r"""Gaussian Copula BIC value

    This function uses a Gaussian copula, with a given parameter of
    dependence "theta", to compute the Bayesian Information Criterion
    (BIC) from a given dataset

    Parameters:
    ----------
    u : numpy.ndarray
        First vector of n elements in [0,1]
    v : numpy.ndarray
        Second vector of n elements in [0,1]
    theta : float
        Parameter of dependence of the Gaussian copula (Pearson's rho)
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
    """Value of the upper tail dependence of the Gaussian copula"""

    return 0


def ltd(theta):
    """Value of the lower tail dependence of the Gaussian copula"""

    return 0
