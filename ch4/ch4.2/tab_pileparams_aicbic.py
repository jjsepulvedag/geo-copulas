# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
AIC/BIC values for the copulas used to fit the Hyperbolic curve-fitting
parameters for piles
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
2011
International journal for numerical and analytical method in geomechan.
Elsevier
2."Characterization of model uncertainty in the static pile design
formula"
Dithinde M et al
2011
Journal of geotechnical and geoenvironmental engineering
Asce
--------------------------------------------------------------------------
"""

# %% ------------------------Import some needed modules------------------------

# Some standar modules
import os
import sys
import numpy as np
import scipy.stats as st

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


# %% -------------------------Fitting copulas to data--------------------------
# List order: [gaussian, student-t, plackett, frank, no16]
# Note: xxx_nu is equal to the number of degrees of freedom of the Student-t
# Driven piles in non-cohesive soils (D-NC)
dnc_theta = np.zeros(5)
dnc_theta[0] = gaussian.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
dnc_theta[1], dnc_nu = student.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
dnc_theta[2] = plackett.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
dnc_theta[3] = frank.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
dnc_theta[4] = no16.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
# Bored piles in non-cohesive soils (B-NC)
bnc_theta = np.zeros(5)
bnc_theta[0] = gaussian.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
bnc_theta[1], bnc_nu = student.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
bnc_theta[2] = plackett.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
bnc_theta[3] = frank.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
bnc_theta[4] = no16.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
# Driven piles in cohesive soils (D-C)
dc_theta = np.zeros(5)
dc_theta[0] = gaussian.fit(vals=(d_c_a, d_c_b), method='moments-t')
dc_theta[1], dc_nu = student.fit(vals=(d_c_a, d_c_b), method='moments-t')
dc_theta[2] = plackett.fit(vals=(d_c_a, d_c_b), method='moments-t')
dc_theta[3] = frank.fit(vals=(d_c_a, d_c_b), method='moments-t')
dc_theta[4] = no16.fit(vals=(d_c_a, d_c_b), method='moments-t')
# Bored piles in cohesive soils (B-C)
bc_theta = np.zeros(5)
bc_theta[0] = gaussian.fit(vals=(b_c_a, b_c_b), method='moments-t')
bc_theta[1], bc_nu = student.fit(vals=(b_c_a, b_c_b), method='moments-t')
bc_theta[2] = plackett.fit(vals=(b_c_a, b_c_b), method='moments-t')
bc_theta[3] = frank.fit(vals=(b_c_a, b_c_b), method='moments-t')
bc_theta[4] = no16.fit(vals=(b_c_a, b_c_b), method='moments-t')

thetas = np.array([dnc_theta, bnc_theta, dc_theta, bc_theta])
nus = np.array([dnc_nu, bnc_nu, dc_nu, bc_nu])


# %% ---------------Estimating AIC/BIC for each copula from data---------------

# Matrix shape: 4rows x 5columns
# Matrix columns order: [gaussian, student-t, plackett, frank, no16]
# Matrix rows order: [D-NC, B-NC, D-C, B-C]

# Empirical distributions of all parameters
rank_dnca = st.rankdata(d_nc_a)/(1 + len(d_nc_a))
rank_dncb = st.rankdata(d_nc_b)/(1 + len(d_nc_b))
rank_bnca = st.rankdata(b_nc_a)/(1 + len(b_nc_a))
rank_bncb = st.rankdata(b_nc_b)/(1 + len(b_nc_b))
rank_dca = st.rankdata(d_c_a)/(1 + len(d_c_a))
rank_dcb = st.rankdata(d_c_b)/(1 + len(d_c_b))
rank_bca = st.rankdata(b_c_a)/(1 + len(b_c_a))
rank_bcb = st.rankdata(b_c_b)/(1 + len(b_c_b))

# Lists of empirical distributions
ranks_a = [rank_dnca, rank_bnca, rank_dca, rank_bca]
ranks_b = [rank_dncb, rank_bncb, rank_dcb, rank_bcb]

# Loops for the AIC and BIC matrixes

aic = np.zeros((4, 5))

for i in range(aic.shape[0]):
    aic[i, 0] = gaussian.aic(ranks_a[i], ranks_b[i], thetas[i, 0])
    aic[i, 1] = student.aic(ranks_a[i], ranks_b[i], thetas[i, 1], nu=nus[i])
    aic[i, 2] = plackett.aic(ranks_a[i], ranks_b[i], thetas[i, 2])
    aic[i, 3] = frank.aic(ranks_a[i], ranks_b[i], thetas[i, 3])
    aic[i, 4] = no16.aic(ranks_a[i], ranks_b[i], thetas[i, 4])

bic = np.zeros((4, 5))

for i in range(bic.shape[0]):
    bic[i, 0] = gaussian.bic(ranks_a[i], ranks_b[i], thetas[i, 0])
    bic[i, 1] = student.bic(ranks_a[i], ranks_b[i], thetas[i, 1], nu=nus[i])
    bic[i, 2] = plackett.bic(ranks_a[i], ranks_b[i], thetas[i, 2])
    bic[i, 3] = frank.bic(ranks_a[i], ranks_b[i], thetas[i, 3])
    bic[i, 4] = no16.bic(ranks_a[i], ranks_b[i], thetas[i, 4])

# %%-----------------Export result tables (AIC and BIC) as .txt----------------

# Export tables in the same folder where this file is
os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.savetxt('aic_copulas_table.txt', aic, fmt='%.3f')
np.savetxt('bic_copulas_table.txt', bic, fmt='%.3f')
np.savetxt('thetas_copulas_table.txt', thetas, fmt='%.3f')

