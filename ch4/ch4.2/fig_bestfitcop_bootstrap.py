# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Copula model uncertainty of Hyperbolic curve-fitting parameters for piles,
by the bootstrap method.
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

Ns = 1_000  # Bootstrap size
N_dnc = len(rank_dnca)
N_bnc = len(rank_bnca)
N_dc  = len(rank_dca)
N_bc  = len(rank_bca)

aic_dnc = np.zeros((5, Ns))
aic_bnc = np.zeros((5, Ns))
aic_dc = np.zeros((5, Ns))
aic_bc = np.zeros((5, Ns))

# Main loop, it fills the aic_vectors of each pile tests
for i in range(Ns):

    # Driven piles - noncohesive soils
    indices = np.random.randint(low=0, high=N_dnc, size=N_dnc)
    sample_a = rank_dnca[(indices)]
    sample_b = rank_dncb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b))
    theta_s, nu_s = student.fit(vals=(sample_a, sample_b))
    theta_p = plackett.fit(vals=(sample_a, sample_b))
    theta_f = frank.fit(vals=(sample_a, sample_b))
    theta_n = no16.fit(vals=(sample_a, sample_b))

    aic_g = gaussian.aic(sample_a, sample_b, theta=theta_g)
    aic_s = student.aic(sample_a, sample_b, theta=theta_s, nu=4)
    aic_p = plackett.aic(sample_a, sample_b, theta=theta_p)
    aic_f = frank.aic(sample_a, sample_b, theta=theta_f)
    aic_n = no16.aic(sample_a, sample_b, theta=theta_n)

    aic_dnc[:, i] = (aic_g, aic_s, aic_p, aic_f, aic_n)

    # Bored Piles - noncohesive soils
    indices = np.random.randint(low=0, high=N_bnc, size=N_bnc)
    sample_a = rank_bnca[(indices)]
    sample_b = rank_bncb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b))
    theta_s, nu_s = student.fit(vals=(sample_a, sample_b))
    theta_p = plackett.fit(vals=(sample_a, sample_b))
    theta_f = frank.fit(vals=(sample_a, sample_b))
    theta_n = no16.fit(vals=(sample_a, sample_b))

    aic_g = gaussian.aic(sample_a, sample_b, theta=theta_g)
    aic_s = student.aic(sample_a, sample_b, theta=theta_s, nu=4)
    aic_p = plackett.aic(sample_a, sample_b, theta=theta_p)
    aic_f = frank.aic(sample_a, sample_b, theta=theta_f)
    aic_n = no16.aic(sample_a, sample_b, theta=theta_n)

    aic_bnc[:, i] = (aic_g, aic_s, aic_p, aic_f, aic_n)

    # Driven Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_dc, size=N_dc)
    sample_a = rank_dca[(indices)]
    sample_b = rank_dcb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b))
    theta_s, nu_s = student.fit(vals=(sample_a, sample_b))
    theta_p = plackett.fit(vals=(sample_a, sample_b))
    theta_f = frank.fit(vals=(sample_a, sample_b))
    theta_n = no16.fit(vals=(sample_a, sample_b))

    aic_g = gaussian.aic(sample_a, sample_b, theta=theta_g)
    aic_s = student.aic(sample_a, sample_b, theta=theta_s, nu=4)
    aic_p = plackett.aic(sample_a, sample_b, theta=theta_p)
    aic_f = frank.aic(sample_a, sample_b, theta=theta_f)
    aic_n = no16.aic(sample_a, sample_b, theta=theta_n)

    aic_dc[:, i] = (aic_g, aic_s, aic_p, aic_f, aic_n)

    # Bored Piles - cohesive soils
    indices = np.random.randint(low=0, high=N_bc, size=N_bc)
    sample_a = rank_bca[(indices)]
    sample_b = rank_bcb[(indices)]

    theta_g = gaussian.fit(vals=(sample_a, sample_b))
    theta_s, nu_s = student.fit(vals=(sample_a, sample_b))
    theta_p = plackett.fit(vals=(sample_a, sample_b))
    theta_f = frank.fit(vals=(sample_a, sample_b))
    theta_n = no16.fit(vals=(sample_a, sample_b))

    aic_g = gaussian.aic(sample_a, sample_b, theta=theta_g)
    aic_s = student.aic(sample_a, sample_b, theta=theta_s, nu=4)
    aic_p = plackett.aic(sample_a, sample_b, theta=theta_p)
    aic_f = frank.aic(sample_a, sample_b, theta=theta_f)
    aic_n = no16.aic(sample_a, sample_b, theta=theta_n)

    aic_bc[:, i] = (aic_g, aic_s, aic_p, aic_f, aic_n)
    print(i)

# %% ---------------------------Plot all the figures---------------------------

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(221)  # Driven piles - noncohesive soils
ax2 = fig.add_subplot(222)  # Bored piles - noncohesive soils
ax3 = fig.add_subplot(223)  # Driven piles - cohesive soils
ax4 = fig.add_subplot(224)  # Bored piles - cohesive soils
axs = [ax1, ax2, ax3, ax4]

titles = ['a', 'b', 'c', 'd']
colors = ['black', 'red', 'blue', 'green', 'orange']
labels = ['Gaussian', 'Student-t', 'Plackett', 'Frank', 'No. 16']
x_limits = np.array([[-60, 20], [-100, 20], [-150, 0], [-150, 0]])
y_limits = np.array([[0, 0.06], [0, 0.035], [0, 0.035], [0, 0.035]])

# Loop to trace the curves
for i in range(len(labels)):
    sns.distplot(aic_dnc[i, :], hist=False, label=labels[i], ax=ax1,
                 color=colors[i])
    sns.distplot(aic_bnc[i, :], hist=False, label=labels[i], ax=ax2,
                 color=colors[i])
    sns.distplot(aic_dc[i, :], hist=False, label=labels[i], ax=ax3,
                 color=colors[i])
    sns.distplot(aic_bc[i, :], hist=False, label=labels[i], ax=ax4,
                 color=colors[i])

# Loop to customize the graphs
for i in range(len(axs)):
    axs[i].set_title('({})'.format(titles[i]), loc='left')
    axs[i].set_xlabel('AIC score')
    axs[i].set_ylabel('Probability density function')
    axs[i].set_xlim(x_limits[i])
    axs[i].set_ylim(y_limits[i])
    axs[i].legend(fontsize='x-small')

ax2.set_xticks(np.linspace(-100, 20, 7))

plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.2/')
fig.savefig('fig_aiccop_bootstrap.pdf')