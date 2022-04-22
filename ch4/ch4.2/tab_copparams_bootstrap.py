# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Mean and Std of the copula dependence parameter of different copulas, 
using the Hyperbolic curve-fitting parameters for piles dataset, and the
bootstrap method.
--------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
--------------------------------------------------------------------------
First version : June 2020
--------------------------------------------------------------------------
Based on:
1."Bivariate simulation using copulas and its applications to
probabilistic pile settlement analysis "
Li DQ et al.
2013
International journal for numerical and analytical method in geomechan.
Elsevier
2."Characterization of model uncertainty in the static pile design
formula"
Dithinde M et al
2011
Journal of geotechnical and geoenvironmental engineering
Asce
3. "Characterization of uncertainty in probabilistic model using bootstrap
method and its application to reliability of piles"
Li DQ et al.
2015
Applied Mathematical Modelling
Elsevier
--------------------------------------------------------------------------
"""

# %% ------------------------Import some needed modules------------------------
# Some standar modules
import os
import sys
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

# Defining the correct path to import pycopulas module and to get data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\..\\')
main_path = os.getcwd()
sys.path.append(main_path)

from pycopulas import gaussian, student, plackett, frank, no16

# %% -------------------------------Import data--------------------------------

file = 'hyperbolic_curve-fitting_parameters_for_piles_Dithinde_etal.txt'
file_path = os.path.join('data', file)
data = np.loadtxt(fname=file_path, delimiter=' ')

# Columns 1 and 2 correspond to driven piles in noncohesive soils
d_nc_a = data[:, 0][np.logical_not(np.isnan(data[:, 0]))]
d_nc_b = data[:, 1][np.logical_not(np.isnan(data[:, 1]))]
# Columns 3 and 4 correspond to bored piles in noncohesive soils
b_nc_a = data[:, 2][np.logical_not(np.isnan(data[:, 2]))]
b_nc_b = data[:, 3][np.logical_not(np.isnan(data[:, 3]))]
# Columns 5 and 6 correspond to driven piles in cohesive soils
d_c_a = data[:, 4][np.logical_not(np.isnan(data[:, 4]))]
d_c_b = data[:, 5][np.logical_not(np.isnan(data[:, 5]))]
# Columns 7 and 8 correspond to bored piles in cohesive soils
b_c_a = data[:, 6][np.logical_not(np.isnan(data[:, 6]))]
b_c_b = data[:, 7][np.logical_not(np.isnan(data[:, 7]))]

# data_list = [d_nc_a, d_c_b, b_nc_a, b_nc_b, d_c_a, d_c_b, b_c_a, b_c_b]

# %% --------------------Define the empirical distributions--------------------

# Empirical distribution for all the tests

rank_dnca = st.rankdata(d_nc_a)/(1 + len(d_nc_a))
rank_dncb = st.rankdata(d_nc_b)/(1 + len(d_nc_b))

rank_bnca = st.rankdata(b_nc_a)/(1 + len(b_nc_a))
rank_bncb = st.rankdata(b_nc_b)/(1 + len(b_nc_b))

rank_dca = st.rankdata(d_c_a)/(1 + len(d_c_a))
rank_dcb = st.rankdata(d_c_b)/(1 + len(d_c_b))

rank_bca = st.rankdata(b_c_a)/(1 + len(b_c_a))
rank_bcb = st.rankdata(b_c_b)/(1 + len(b_c_b))

# %% ---------------------------Bootstrap procedure----------------------------

Ns = 10_000  # Bootstrap size
N_dnc = len(rank_dnca)
N_bnc = len(rank_bnca)
N_dc  = len(rank_dca)
N_bc  = len(rank_bca)


theta_dnc = np.zeros((Ns, 4))
theta_bnc = np.zeros((Ns, 4))
theta_dc = np.zeros((Ns, 4))
theta_bc = np.zeros((Ns, 4))

# Main loop, it fills the aic_vectors of each pile tests
for i in range(Ns):

    # Driven piles - noncohesive soils
    indices = np.random.randint(low=0, high=N_dnc, size=N_dnc)
    sample_a = rank_dnca[(indices)]
    sample_b = rank_dncb[(indices)]

    theta_g = gaussian.fit(sample_a, sample_b)
    theta_p = plackett.fit(sample_a, sample_b)
    theta_f = frank.fit(sample_a, sample_b)
    theta_n = no16.fit(sample_a, sample_b)

    theta_dnc[i] = (theta_g, theta_p, theta_f, theta_n)

    # Bored Piles - noncohesive soils
    indices = np.random.randint(low=0, high=N_bnc, size=N_bnc)
    sample_a = rank_bnca[(indices)]
    sample_b = rank_bncb[(indices)]

    theta_g = gaussian.fit(sample_a, sample_b)
    theta_p = plackett.fit(sample_a, sample_b)
    theta_f = frank.fit(sample_a, sample_b)
    theta_n = no16.fit(sample_a, sample_b)

    theta_bnc[i] = (theta_g, theta_p, theta_f, theta_n)

    # Driven Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_dc, size=N_dc)
    sample_a = rank_dca[(indices)]
    sample_b = rank_dcb[(indices)]

    theta_g = gaussian.fit(sample_a, sample_b)
    theta_p = plackett.fit(sample_a, sample_b)
    theta_f = frank.fit(sample_a, sample_b)
    theta_n = no16.fit(sample_a, sample_b)

    theta_dc[i] = (theta_g, theta_p, theta_f, theta_n)

    # Bored Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_bc, size=N_bc)
    sample_a = rank_bca[(indices)]
    sample_b = rank_bcb[(indices)]

    theta_g = gaussian.fit(sample_a, sample_b)
    theta_p = plackett.fit(sample_a, sample_b)
    theta_f = frank.fit(sample_a, sample_b)
    theta_n = no16.fit(sample_a, sample_b)

    theta_bc[i] = (theta_g, theta_p, theta_f, theta_n)

# %% ----------------Compute and save the arrays means and stds----------------

# Matrix arrangement
# Columns: Gaussian, Plackett, Frank, No 16
# Rows   : D-NC, B-NC, D-C, B-C

means = np.zeros((4, 4))
stds = np.zeros((4, 4))
covs = np.zeros((4, 4))

thetas = [theta_dnc, theta_bnc, theta_dc, theta_bc]

for i in range(means.shape[0]):
    for j in range(means.shape[1]):
        means[i, j] = np.mean(thetas[i][:, j])
        stds[i, j] = np.std(thetas[i][:, j])
        covs[i, j] = st.variation(thetas[i][:, j])


# %% ------------------------Export both tables as .txt------------------------

# Export tables in the same folder where this file is
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.savetxt('theta_mean_table.txt', means, fmt='%.3f')
np.savetxt('theta_std_table.txt', stds, fmt='%.3f')
np.savetxt('theta_cov_table.txt', covs, fmt='%.3f')
