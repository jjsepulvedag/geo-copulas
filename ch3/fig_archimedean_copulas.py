# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of some archimedean copulas with different marginals:
- Frank copula
- Nelsen No 16 copula
- Gumbel copula
- Clayton copula
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

# Defining the correct path to import pycopulas module
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\')
main_path = os.getcwd()
sys.path.append(main_path)
from pycopulas import frank, no16, gumbel, clayton

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

# %% -------------Get the vectors of gaussian and student copulas--------------

# Copula parameter of dependence corresponding to a Kendall tau of  0.3
theta_frank = 2.917
theta_no16 = 3.723
theta_gumbe = 1.428
theta_clayt = 0.857

# Frank copula
Fxy_frank = frank.pdf(u=u, v=v, theta=theta_frank)*f*g

# No16 copula
Fxy_no16 = no16.pdf(u=u, v=v, theta=theta_no16)*f*g

# Gumbel copula
Fxy_gumbe = gumbel.pdf(u=u, v=v, theta=theta_gumbe)*f*g

# Clayton copula
Fxy_clayton = clayton.pdf(u=u, v=v, theta=theta_clayt)*f*g

# %%-----------------------------Plot the figures------------------------------

fig = plt.figure(figsize=(7, 7))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

axs = [ax1, ax2, ax3, ax4]

Fxy_copulas = [Fxy_frank, Fxy_no16, Fxy_gumbe, Fxy_clayton]
names = ['a', 'b', 'c', 'd']

levels = np.linspace(0, 0.20, 7)
ticks = np.linspace(-3, 3, 7)


for i in range(len(axs)):
    cs = axs[i].contour(X, Y, Fxy_copulas[i], levels=levels, colors='black')
    axs[i].clabel(cs, inline=1, fontsize=8)
    axs[i].set_yticks(ticks)
    axs[i].set_xticks(ticks)
    axs[i].set_xlabel(r'$x$')
    axs[i].set_ylabel(r'$y$')
    axs[i].grid(linestyle='--')
    axs[i].set_title('({})'.format(names[i]), loc='left')

plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../2.0_figures/ch3')
fig.savefig('fig_contour_archimedean.pdf')

