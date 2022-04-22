# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Contour plot of the Plackett copula
--------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
--------------------------------------------------------------------------
First version : September 2020
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

# Defining the correct path to import pycopulas module
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\')
main_path = os.getcwd()
sys.path.append(main_path)

from pycopulas import plackett

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# %% ------------------Define the base grid and the marginals------------------

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

X, Y = np.meshgrid(x, y)

# For simplicity, both marginals are standard normal
u = st.norm.cdf(X)
v = st.norm.cdf(Y)
f = st.norm.pdf(X)
g = st.norm.pdf(Y)

# %% --------------------Get the vectors of Plackett copula--------------------

# Copula parameter of dependence corresponding to a Kendall tau of 0.3
tau = 0.3
theta_plack = plackett.fit(corr=0.3, method='moments-t')

# Plackett copula
Fxy_plack = plackett.pdf(u=u, v=v, theta=theta_plack)*f*g

# %%-----------------------------Plot the figures------------------------------

fig = plt.figure(figsize=(3, 3))
ax1 = fig.add_subplot(111)

levels = np.linspace(0, 0.20, 7)
ticks = np.linspace(-3, 3, 7)

# Plackett copula
cs1 = ax1.contour(X, Y, Fxy_plack, levels=levels, colors='black')
# ax1.contour(X, Y, Fxy_gaussian, levels=levels, colors='black')
ax1.clabel(cs1, inline=1, fontsize=8)
ax1.set_yticks(ticks)
ax1.set_xticks(ticks)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.grid(linestyle='--')
plt.tight_layout()


# plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../2.0_figures/ch3')
fig.savefig('fig_contour_plackett.pdf')
