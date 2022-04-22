# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering.
Graphs of simulated c-f from a Consolidated Drained triaxial test in the
Xiaolangdi hydropower station
--------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
--------------------------------------------------------------------------
First version : June 2020
--------------------------------------------------------------------------
Based on:
1."Modelling dependence with Copulas, and applications to risk management"
Embrechts Paul et al.
2001
Department of mathematics, ETHZ
2."Risk and reliability in geotechnical engineering"
Phoon KK, Ching J
2014
CRC Press
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

from pycopulas import gaussian, student, plackett, frank, no16, clayton

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# %% -------------------------------Import data--------------------------------

file_path = os.path.join('data', 'xiaolangdi_hydropower.txt')
data = np.loadtxt(fname=file_path, delimiter=' ')

# Columns 1 and 2 correspond to CD triaxial tests
cdc = data[:, 1][np.logical_not(np.isnan(data[:, 1]))]
cdp = data[:, 2][np.logical_not(np.isnan(data[:, 2]))]

# Adjusting marginals for c and phi by Maximum likelihood estimation
abet, bbet, locbet, scalebet = st.beta.fit(cdc, floc=0, fscale=120)
slog, loclog, scalelog = st.lognorm.fit(cdp, floc=0)

# Defining the marginals 
coh_beta = st.beta(a=abet, b=bbet, loc=locbet, scale=scalebet)
phi_logn = st.lognorm(s=slog, loc=loclog, scale=scalelog)

# %% ------------------Random sampling from different copulas------------------
# Gaussian copula 
theta_gauss = gaussian.fit(vals=(cdc, cdp))
u, v = gaussian.rvs(theta_gauss, 300)
cdc_gauss = coh_beta.ppf(u)
cdp_gauss = phi_logn.ppf(v)
# Student-t copula 
theta_stud, nu_stud = student.fit(vals=(cdc, cdp))
u, v = student.rvs(theta=theta_stud, nu=nu_stud, n=300)
cdc_stud = coh_beta.ppf(u)
cdp_stud = phi_logn.ppf(v)
# Plackett copula
theta_plac = plackett.fit(vals=(cdc, cdp))
u, v = plackett.rvs(theta=theta_plac, n=300)
cdc_plac = coh_beta.ppf(u)
cdp_plac = phi_logn.ppf(v)
# Frank copula
theta_frank = frank.fit(vals=(cdc, cdp))
u, v = frank.rvs(theta=theta_frank, n=300)
cdc_frank = coh_beta.ppf(u)
cdp_frank = phi_logn.ppf(v)
# No16 copula
theta_no16 = no16.fit(vals=(cdc, cdp))
u, v = no16.rvs(theta=theta_no16, n=300)
cdc_no16 = coh_beta.ppf(u)
cdp_no16 = phi_logn.ppf(v)
# Clayton copula
theta_clay = clayton.fit(cdc, cdp)
u, v = clayton.rvs(theta=theta_clay, n=300)
cdc_clay = coh_beta.ppf(u)
cdp_clay = phi_logn.ppf(v)

# %% --------------Plot the random sampling vs the original data---------------

fig = plt.figure(figsize=(7, 7))
gs = gs.GridSpec(nrows=3, ncols=4)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:])
ax5 = plt.subplot(gs[2, 1:3])

# Define some needed lists 
axes = [ax1, ax2, ax3, ax4, ax5]
cdc_sim = [cdc_gauss, cdc_stud, cdc_plac, cdc_frank, cdc_no16]
cdp_sim = [cdp_gauss, cdp_stud, cdp_plac, cdp_frank, cdp_no16]
graph_names = ['a', 'b', 'c', 'd', 'e']

for i in range(len(axes)):
    axes[i].scatter(cdc_sim[i], cdp_sim[i], facecolors='none', edgecolors='r',
                    label='Simulated')
    axes[i].scatter(cdc, cdp, color='k', label='Measured')
    axes[i].set_xlabel(r'c$\left(KPa\right)$')
    axes[i].set_ylabel(r'$\phi\left(^\circ\right)$')
    axes[i].set_title('({})'.format(graph_names[i]), loc='left', fontsize=13)
    axes[i].set_xlim((0, 120))
    axes[i].set_ylim((15, 35))
    axes[i].legend()
    axes[i].grid()
    axes[i].set_axisbelow(True)


plt.tight_layout()
# plt.show()


# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.1/')
fig.savefig('fig_simulation_cf.pdf')
