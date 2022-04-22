# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of Hyperbolic curve-fitting parameters for piles
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
import matplotlib as mpl
import matplotlib.pyplot as plt

# Defining the correct path to import pycopulas module and to get data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\..\\')
main_path = os.getcwd()
sys.path.append(main_path)

from pycopulas import gaussian, student, plackett, frank, no16

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

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

data_list = [d_nc_a, d_c_b, b_nc_a, b_nc_b, d_c_a, d_c_b, b_c_a, b_c_b]

# %% ----------Fit lognormal marginal distribution for each parameter----------

# params: array where each row contains the fit parameters of each data set
params = np.zeros((8, 3))

for i in range(params.shape[0]):
    params[i] = st.lognorm.fit(data_list[i], floc=0)

# dists: list that contains the fitted lognormal distribution of each data set
dists = []

for i in range(len(params)):
    x = st.lognorm(s=params[i, 0], loc=params[i, 1], scale=params[i, 2])
    dists.append(x)

# %% --------------Simulate "a" and "b" as independent variables---------------

ind_dnca = dists[0].ppf(st.uniform.rvs(loc=0, scale=1, size=len(d_nc_a)))
ind_dncb = dists[1].ppf(st.uniform.rvs(loc=0, scale=1, size=len(d_nc_b)))

ind_bnca = dists[2].ppf(st.uniform.rvs(loc=0, scale=1, size=len(b_nc_a)))
ind_bncb = dists[3].ppf(st.uniform.rvs(loc=0, scale=1, size=len(b_nc_b)))

ind_dca = dists[4].ppf(st.uniform.rvs(loc=0, scale=1, size=len(d_c_a)))
ind_dcb = dists[5].ppf(st.uniform.rvs(loc=0, scale=1, size=len(d_c_b)))

ind_bca = dists[6].ppf(st.uniform.rvs(loc=0, scale=1, size=len(b_c_a)))
ind_bcb = dists[7].ppf(st.uniform.rvs(loc=0, scale=1, size=len(b_c_b)))

# %% -------------------------Simulation from copulas--------------------------
# Driven piles in non-cohesive soils: Best fit copula -> Gaussian copula
dnc_theta = gaussian.fit(vals=(d_nc_a, d_nc_b), method='moments-t')
u_dnca, v_dncb = gaussian.rvs(p=dnc_theta, n=len(d_nc_a))
sim_dnca = dists[0].ppf(u_dnca)
sim_dncb = dists[1].ppf(v_dncb)
# Bored piles in non-cohesive soils: Best fit copula -> Plackett copula
bnc_theta = plackett.fit(vals=(b_nc_a, b_nc_b), method='moments-t')
u_bnca, v_bncb = plackett.rvs(theta=bnc_theta, n=len(b_nc_a))
sim_bnca = dists[2].ppf(u_bnca)
sim_bncb = dists[3].ppf(v_bncb)
# Driven piles in cohesive soils: Best fit copula -> Frank copula
dc_theta = plackett.fit(vals=(d_c_a, d_c_b), method='moments-t')
u_dca, v_dcb = frank.rvs(theta=dc_theta, n=len(d_c_a))
sim_dca = dists[4].ppf(u_dca)
sim_dcb = dists[5].ppf(v_dcb)
# Bored piles in cohesive soils: Best fit copula -> Plackett copula
bc_theta = plackett.fit(vals=(b_c_a, b_c_b), method='moments-t')
u_bca, v_bcb = plackett.rvs(theta=bc_theta, n=len(b_c_a))
sim_bca = dists[6].ppf(u_bca)
sim_bcb = dists[7].ppf(v_bcb)

# %% -------Plot graphs with the simulated data from the best fit copula-------

width, height = 12, 3

# s: settlement -> axis "x"
s = np.linspace(0, 30, 1000)

# Figure 1 corresponds to the driven piles in non-cohesive soils
fig1 = plt.figure(figsize=(width, height))
ax11 = fig1.add_subplot(131)
ax12 = fig1.add_subplot(132)
ax13 = fig1.add_subplot(133)

axes1 = [ax11, ax12, ax13]

for i in range(len(d_nc_a)):
    q = s/(d_nc_a[i] + d_nc_b[i]*s)
    p = s/(sim_dnca[i] + sim_dncb[i]*s)
    r = s/(ind_dnca[i] + ind_dncb[i]*s)
    ax11.plot(s, q, color='k', linewidth=0.5)
    ax12.plot(s, p, color='k', linewidth=0.5)
    ax13.plot(s, r, color='k', linewidth=0.5)


# Figure 2 corresponds to bored piles in non-cohesive soils
fig2 = plt.figure(figsize=(width, height))
ax21 = fig2.add_subplot(131)
ax22 = fig2.add_subplot(132)
ax23 = fig2.add_subplot(133)

axes2 = [ax21, ax22, ax23]

for i in range(len(b_nc_a)):
    q = s/(b_nc_a[i] + b_nc_b[i]*s)
    p = s/(sim_bnca[i] + sim_bncb[i]*s)
    r = s/(ind_bnca[i] + ind_bncb[i]*s)
    ax21.plot(s, q, color='k', linewidth=0.5)
    ax22.plot(s, p, color='k', linewidth=0.5)
    ax23.plot(s, r, color='k', linewidth=0.5)


# Figure 3 corresponds to driven piles in cohesive soils
fig3 = plt.figure(figsize=(width, height))
ax31 = fig3.add_subplot(131)
ax32 = fig3.add_subplot(132)
ax33 = fig3.add_subplot(133)

axes3 = [ax31, ax32, ax33]

for i in range(len(d_c_a)):
    q = s/(d_c_a[i] + d_c_b[i]*s)
    p = s/(sim_dca[i] + sim_dcb[i]*s)
    r = s/(ind_dca[i] + ind_dcb[i]*s)
    ax31.plot(s, q, color='k', linewidth=0.5)
    ax32.plot(s, p, color='k', linewidth=0.5)
    ax33.plot(s, r, color='k', linewidth=0.5)


# Figure 4 corresponds to bored piles in cohesive soils
fig4 = plt.figure(figsize=(width, height))
ax41 = fig4.add_subplot(131)
ax42 = fig4.add_subplot(132)
ax43 = fig4.add_subplot(133)

axes4 = [ax41, ax42, ax43]

for i in range(len(b_c_a)):
    q = s/(b_c_a[i] + b_c_b[i]*s)
    p = s/(sim_bca[i] + sim_bcb[i]*s)
    r = s/(ind_bca[i] + ind_bcb[i]*s)
    ax41.plot(s, q, color='k', linewidth=0.5)
    ax42.plot(s, p, color='k', linewidth=0.5)
    ax43.plot(s, r, color='k', linewidth=0.5)        


# Final loop to edit configurations in axis presentations
axes = np.array([axes1, axes2, axes3, axes4])
names = [r'$\left(a\right)$', r'$\left(b\right)$', r'$\left(c\right)$']

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i, j].set_xlabel(r'$s\left(mm\right)$')
        axes[i, j].set_ylabel(r'$\frac{Q}{Q_{STC}}$')
        axes[i, j].set_xlim((0, 30))
        axes[i, j].set_ylim((0, 1.5))
        axes[i, j].set_title('{}'.format(names[j]), loc='left', fontsize=22)

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.2/')

# fig1.savefig('fig_dnc_curves.pdf')
# fig2.savefig('fig_bnc_curves.pdf')
fig3.savefig('fig_dc_curves.pdf')
# fig4.savefig('fig_bc_curves.pdf')
