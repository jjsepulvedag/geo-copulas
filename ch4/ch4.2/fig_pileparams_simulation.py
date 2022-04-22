# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graphs of simulated 'a' and 'b' parameters from a driven pile test
in cohesive soils. 
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
import matplotlib.gridspec as gs

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

# %% ----------Adjust Lognormal dist for both parameters of D-C test-----------

s_a, loc_a, scale_a = st.lognorm.fit(d_c_a, floc=0)
s_b, loc_b, scale_b = st.lognorm.fit(d_c_b, floc=0)

# Defining the marginals 
a_logn = st.lognorm(s=s_a, loc=loc_a, scale=scale_a)
b_logn = st.lognorm(s=s_b, loc=loc_b, scale=scale_b)

# %% -----------------------Random sampling from copulas-----------------------
# Gaussian copula 
theta_gauss = gaussian.fit(vals=(d_c_a, d_c_b), method='moments-t')
u, v = gaussian.rvs(theta_gauss, 300)
dca_gauss = a_logn.ppf(u)
dcb_gauss = b_logn.ppf(v)
# Student-t copula 
theta_stud, nu_stud = student.fit(vals=(d_c_a, d_c_b), method='moments-t')
u, v = student.rvs(theta=theta_stud, nu=nu_stud, n=300)
dca_stud = a_logn.ppf(u)
dcb_stud = b_logn.ppf(v)
# Plackett copula
theta_plac = plackett.fit(vals=(d_c_a, d_c_b), method='moments-t')
u, v = plackett.rvs(theta=theta_plac, n=300)
dca_plac = a_logn.ppf(u)
dcb_plac = b_logn.ppf(v)
# Frank copula
theta_frank = frank.fit(vals=(d_c_a, d_c_b), method='moments-t')
u, v = frank.rvs(theta=theta_frank, n=300)
dca_frank = a_logn.ppf(u)
dcb_frank = b_logn.ppf(v)
# No16 copula
theta_no16 = no16.fit(vals=(d_c_a, d_c_b), method='moments-t')
u, v = no16.rvs(theta=theta_no16, n=300)
dca_no16 = a_logn.ppf(u)
dcb_no16 = b_logn.ppf(v)

sim_data = np.array([[dca_gauss, dcb_gauss], 
                     [dca_stud, dcb_stud],
                     [dca_plac, dcb_plac],
                     [dca_frank, dcb_frank],
                     [dca_no16, dcb_no16]])

# %% -------------------Plot the simulated data from copulas-------------------

fig = plt.figure(figsize=(7, 7))
gs = gs.GridSpec(nrows=3, ncols=4)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:])
ax5 = plt.subplot(gs[2, 1:3])

axes = [ax1, ax2, ax3, ax4, ax5]
names = ['a', 'b', 'c', 'd', 'e']

for i in range(len(axes)):
    axes[i].scatter(sim_data[i, 0], sim_data[i, 1], facecolor='none', 
                    edgecolor='r', label='Simulated')
    axes[i].scatter(d_c_a, d_c_b, color='k', label='Measured')
    axes[i].set_xlim((0, 16))
    axes[i].set_ylim((0.4, 1.2))
    axes[i].set_title('({})'.format(names[i]), loc='left')
    axes[i].set_xlabel('a')
    axes[i].set_ylabel('b')
    axes[i].legend()

plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.2/')
fig.savefig('fig_pileparams_simulation.pdf')

