# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of elliptical copulas (Gaussian/Student) with different marginals
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
from pycopulas import gaussian, student

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

# Copula parameter of dependence corresponding to a Kendall tau of 0.3
tau = 0.3
theta_gauss = np.sin(tau*np.pi/2)
theta_studt = np.sin(tau*np.pi/2)

# Gaussian copula
Fxy_gaussian = gaussian.pdf(u=u, v=v, theta=theta_gauss)*f*g

# Student copula (3 degrees of freedom)
Fxy_student = student.pdf(u=u, v=v, theta=theta_studt, nu=3)*f*g

# %%-----------------------------Plot the figures------------------------------

fig = plt.figure(figsize=(7, 3.5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

levels = np.linspace(0, 0.20, 7)
ticks = np.linspace(-3, 3, 7)

# Gaussian copula
cs1 = ax1.contour(X, Y, Fxy_gaussian, levels=levels, colors='black')
# ax1.contour(X, Y, Fxy_gaussian, levels=levels, colors='black')
ax1.clabel(cs1, inline=1, fontsize=8)
ax1.set_yticks(ticks)
ax1.set_xticks(ticks)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.grid(linestyle='--')
ax1.set_title(r'$\left(a\right)$', loc='left')

# Student-t copula
cs2 = ax2.contour(X, Y, Fxy_student, levels=levels[:-1], colors='black')
# ax2.contour(X, Y, Fxy_student, levels=levels, colors='black')
ax2.clabel(cs2, inline=1, fontsize=8)
ax2.set_yticks(ticks)
ax2.set_xticks(ticks)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.grid(linestyle='--')
ax2.set_title(r'$\left(b\right)$', loc='left')

plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../2.0_figures/ch3')
fig.savefig('fig_contour_elliptical.pdf')
