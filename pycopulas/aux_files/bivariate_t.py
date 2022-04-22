# -*- coding: utf-8 -*-
r"""Bivariate Student-t distribution module

This module computes the PDF or CDF of a given bivariate Student-t
distribution

Functions:
----------
pdf : calculates the pdf value of a bivariate Student-t distribution
cdf : calculates the cdf value of a bivariate Student-t distribution

Notes:
----------
This module is restricted to a bivariate case of the Student-t 
distribution. Future work should include extensions to more than two
dimensions.

References:
----------
1. https://en.wikipedia.org/wiki/Multivariate_t-distribution
2. "Comparison of Methods for the Computation of Multivariate t 
Probabilities"
Genz A. and Bretz F.
2002
Journal of Computational and Graphical Statistics
3. "A bivariate generalization of Student's t-distribution with tables for
certain special cases"
Dunnett, C.W. and M. Sobel
1954
Biometrika
"""


import numpy as np
import scipy.stats as st
from scipy.special import gamma
from scipy.linalg import det, inv


def pdf(u, v, theta, nu):
    """PDF value of a given Student-t bivariate distribution

    This functions computes the PDF value of a given Bivariate Student-t
    distribution.

    Parameters:
    ----------
    u : float
        Value of the first variable
    v : float
        Value of the second variable
    theta : float
        pearson's rho correlation coefficient between two random variables
    nu : int (>=1)
        Degrees of freedom
    Returns:
    ----------
    pdf : float
        PDF value of a given Student-t distribution, obtained from a pair
        of realizations u and v
    Notes:
    ----------
    For space purposes, the whole function was split in two factors,
    i.e. f1 and f2.
    """

    p = 2  # Dimension (restricted to a bivariate case)
    cm = np.array([[1, theta], [theta, 1]])  # Correlation matrix
    S = inv(cm)  # Inverse of the matrix of correlation
    x = np.array([u, v])  # Vector of random variables

    # Matrix multiplication (x*S*x)
    m = np.array([x[0]*S[0, 0] + x[1]*S[1, 0], x[0]*S[0, 1] + x[1]*S[1, 1]])
    matmulti = x[0]*m[0] + x[1]*m[1]

    # Factor 1 and 2 (f1 and f2) used to compute the pdf
    f1 = gamma((nu+p)/2)/(gamma(nu/2)*nu**(p/2)*np.pi**(p/2)*det(cm)**(1/2))
    f2 = (1 + (1/nu)*(matmulti))**(-(nu+p)/2)

    pdf = f1*f2

    return pdf


def cdf(dh, dk, r, nu):
    """CDF value of a given Student-t bivariate distribution

    This functions computes the CDF value of a given Bivariate Student-t
    distribution.

    Parameters:
    ----------
    dh : float
        1st upper integration limit 
    dk : float
        2nd upper integration limit
    r : float
        pearson's rho correlation coefficient between two random variables
    nu : int (>=1)
        Degrees of freedom
    Returns:
    ----------
    p : float
        CDF value of a given Student-t distribution, obtained from a pair
        of realizations u and v
    Notes:
    ----------
    As there is no analytical expression to compute the CDF of a bivariate
    Student-t distribution, and approximation must be used. In this way,
    an algorithm, developed by Alan Genz in MatLab, was adapted to Python
    see "http://www.math.wsu.edu/faculty/genz/software/matlab/bvtl.m" for
    the MatLab algorithm. This algorithm in turn is based on the work
    done by Dunnett, C.W. and M. Sobel (1954)

    All the structure and terminology of the Genz's algorithm was 
    preserved, in order to facilitate comparisons and corrections
    """

    corr = np.array([[1, r], [r, 1]])
    if nu < 1:
        p = st.multivariate_normal.cdf([dh, dk], mean=None, cov=corr)
        return p
    elif dh == -np.inf or dk == -np.inf:
        p = 0
        return p
    elif dh == np.inf:
        if dk == np.inf:
            p = 1
            return p
        else:
            p = st.t.cdf(dk, df=nu)
            return p
    elif dk == np.inf:
        p = st.t.cdf(dh, df=nu)
        return p
    elif 1 - r < np.finfo(float).eps:
        p = st.t.cdf(np.min([dh, dk]), nu)
    elif r + 1 < np.finfo(float).eps:
        p = 0
        if dh > -dk:
            p = st.t.cdf(dh, nu) - st.t.cdf(-dk, nu)
        return p
    else: 
        tpi = 2*np.pi
        ors = 1 - r*r
        hrk = dh - r*dk
        krh = dk - r*dh
        if np.abs(hrk) + ors > 0:
            xnhk = hrk**2/(hrk**2 + ors*(nu + dk**2))
            xnkh = krh**2/(krh**2 + ors*(nu + dh**2))
        else: 
            xnhk = 0
            xnkh = 0
        hs = np.sign(dh - r*dk)
        ks = np.sign(dk - r*dh)
        if nu % 2 == 0:
            bvt = np.arctan2(np.sqrt(ors), -r)/tpi
            gmph = dh/np.sqrt(16*(nu + dh**2))
            gmpk = dk/np.sqrt(16*(nu + dk**2))
            btnckh = 2*np.arctan2(np.sqrt(xnkh), np.sqrt(1 - xnkh))/np.pi
            btpdkh = 2*np.sqrt(xnkh*(1 - xnkh))/np.pi
            btnchk = 2*np.arctan2(np.sqrt(xnhk), np.sqrt(1 - xnhk))/np.pi
            btpdhk = 2*np.sqrt(xnhk*(1 - xnhk))/np.pi
            for j in range(1, int(nu/2) + 1):
                bvt = bvt + gmph*(1 + ks*btnckh)
                bvt = bvt + gmpk*(1 + hs*btnchk)
                btnckh = btnckh + btpdkh 
                btpdkh = 2*j*btpdkh*(1 - xnkh)/(2*j + 1)
                btnchk = btnchk + btpdhk 
                btpdhk = 2*j*btpdhk*(1 - xnhk)/(2*j + 1)
                gmph = gmph*(j - 1/2)/(j*(1 + dh**2/nu))
                gmpk = gmpk*(j - 1/2)/(j*(1 + dk**2/nu))
        else:
            qhrk = np.sqrt(dh**2 + dk**2 - 2*r*dh*dk + nu*ors)
            hkrn = dh*dk + r*nu
            hkn = dh*dk - nu
            hpk = dh + dk
            bvt = np.arctan2(-np.sqrt(nu)*(hkn*qhrk + hpk*hkrn), hkn*hkrn - \
                  nu*hpk*qhrk)/tpi
            if bvt < -10*np.finfo(float).eps:
                bvt = bvt + 1
            gmph = dh/(tpi*np.sqrt(nu)*(1 + dh**2/nu))
            gmpk = dk/(tpi*np.sqrt(nu)*(1 + dk**2/nu))
            btnckh = np.sqrt(xnkh)
            btpdkh = btnckh
            btnchk = np.sqrt(xnhk)
            btpdhk = btnchk
            for j in range(1, int((nu - 1)/2) + 1):
                bvt = bvt + gmph*(1 + ks*btnckh)
                bvt = bvt + gmpk*(1 + hs*btnchk)
                btpdkh = (2*j-1)*btpdkh*(1 - xnkh)/(2*j)
                btnckh = btnckh + btpdkh
                btpdhk = (2*j-1)*btpdhk*(1 - xnhk)/(2*j)
                btnchk = btnchk + btpdhk
                gmph = gmph*j/((j + 1/2)*(1 + dh**2/nu))
                gmpk = gmpk*j/((j + 1/2)*(1 + dk**2/nu))
        p = bvt
    return p
