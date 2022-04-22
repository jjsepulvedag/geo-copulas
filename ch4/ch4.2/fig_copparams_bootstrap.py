# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Uncertainty of the copula parameter of dependence of Hyperbolic
curve-fitting parameters for piles, by the bootstrap method.
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

# %% -----------------------Importing some needed modules----------------------
# Some standar modules
import os
import sys
import numpy as np
import seaborn as sns
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

Ns = 50  # Bootstrap size
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

    theta_g = gaussian.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_p = plackett.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_f = frank.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_n = no16.fit(vals=(sample_a, sample_b), method='moments-t')

    theta_dnc[i] = (theta_g, theta_p, theta_f, theta_n)
    print(i, 'a')
    # Bored Piles - noncohesive soils
    indices = np.random.randint(low=0, high=N_bnc, size=N_bnc)
    sample_a = rank_bnca[(indices)]
    sample_b = rank_bncb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_p = plackett.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_f = frank.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_n = no16.fit(vals=(sample_a, sample_b), method='moments-t')

    theta_bnc[i] = (theta_g, theta_p, theta_f, theta_n)
    print(i, 'b')
    # Driven Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_dc, size=N_dc)
    sample_a = rank_dca[(indices)]
    sample_b = rank_dcb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_p = plackett.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_f = frank.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_n = no16.fit(vals=(sample_a, sample_b), method='moments-t')

    theta_dc[i] = (theta_g, theta_p, theta_f, theta_n)
    print(i, 'c')
    # Bored Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_bc, size=N_bc)
    sample_a = rank_bca[(indices)]
    sample_b = rank_bcb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_p = plackett.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_f = frank.fit(vals=(sample_a, sample_b), method='moments-t')
    theta_n = no16.fit(vals=(sample_a, sample_b), method='moments-t')

    theta_bc[i] = (theta_g, theta_p, theta_f, theta_n)
    print(i, 'd')

# %% ---------------------------Plot all the figures---------------------------

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(221)  # Driven piles - noncohesive soils
ax2 = fig.add_subplot(222)  # Bored piles - noncohesive soils
ax3 = fig.add_subplot(223)  # Driven piles - cohesive soils
ax4 = fig.add_subplot(224)  # Bored piles - cohesive soils
axs = [ax1, ax2, ax3, ax4]

titles = ['a', 'b', 'c', 'd']
x_limits = np.array([[-1, -0.3], [0, 0.3], [-50, 0], [0, 0.04]])

for i in range(len(axs)):
    sns.kdeplot(theta_dnc[:, i], label='D-NC', ax=axs[i], color='k')
    sns.kdeplot(theta_bnc[:, i], label='B-NC', ax=axs[i], color='r')
    sns.kdeplot(theta_dc[:, i], label='D-C', ax=axs[i], color='b')
    sns.kdeplot(theta_bc[:, i], label='B-C', ax=axs[i], color='g')
    axs[i].set_title('({})'.format(titles[i]), loc='left')
    axs[i].set_xlabel(r'Copula parameter of dependence $\theta$')
    axs[i].set_ylabel('Probability density function')
    axs[i].set_xlim(x_limits[i])

plt.tight_layout()
plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.2/')
fig.savefig('fig_copparams_bootstrap.pdf')