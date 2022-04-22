# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of lower/upper bound, and independent copula.
--------------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
-------------------------------------------------------------------------------
First version : April 2020
-------------------------------------------------------------------------------
Based on:
1."An introduction to copulas"
   Roger B. Nelsen.
   2006
   Springer Science & Business Media.
2."Modelling dependence with copulas and applications to risk management"
   Paul Embrechts et al.
   2001
   Rapport technique, Département de mathématiques, Institut Fédéral de 
   Technologie de Zurich, volume 14.
-------------------------------------------------------------------------------
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D
plt.close('all')

# %% ---------------------------Some previous steps----------------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# %% -----------------------Define some needed functions-----------------------


# Function for the lower limit W(u,v) = max(u+v-1,0)
def func_w(x, y):
    return np.maximum(x + y - 1, 0)


# Function for the upper limit M(u,v) = min(u,v)
def func_m(x, y):
    return np.minimum(x, y)


# Function for the independent copula P(u,v) = u*v
def func_p(x, y):
    return x*y


# %% ----------------------------- MAIN CODE ----------------------------------


# Functions domain
u = np.linspace(0, 1, 10)
v = np.linspace(0, 1, 10)
U, V = np.meshgrid(u, v)

# Functions range
W = func_w(U, V)
M = func_m(U, V)
P = func_p(U, V)


# %% ------------------------- GRAPHING RESULTS -------------------------------

fig = plt.figure(figsize=(9.4, 4.7))


ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

ax1.plot_wireframe(U, V, W, color='k')
ax1.view_init(elev=15, azim=-135)
# ax1.set_title(r'$W\left(u,v\right) = \left(u + v - 1, 0\right)$')
ax1.set_xlabel(r'$F\left(X\right) = u$')
ax1.set_ylabel(r'$G\left(Y\right) = v$')
ax1.zaxis.set_rotate_label(False)  # For being able to rotate Z axis
ax1.set_zlabel(r'$W\left(u,v\right)$', rotation=90)
ax1.set_title(r'$\left(a\right)$', loc='left')
ax1.set_xticks(np.linspace(0, 1, 3))
ax1.set_yticks(np.linspace(0, 1, 3))
ax1.set_zticks(np.linspace(0, 1, 3))
# ax.grid(False)

ax2.plot_wireframe(U, V, P, color='k')
ax2.view_init(elev=15, azim=-135)
# ax2.set_title(r'$\prod\left(u,v\right) = uv$')
ax2.set_xlabel(r'$F\left(X\right) = u$')
ax2.set_ylabel(r'$G\left(Y\right) = v$')
ax2.zaxis.set_rotate_label(False)  # For being able to rotate Z axis
ax2.set_zlabel(r'$\prod\left(u,v\right)$', rotation=90)
ax2.set_title(r'$\left(b\right)$', loc='left')
ax2.set_xticks(np.linspace(0, 1, 3))
ax2.set_yticks(np.linspace(0, 1, 3))
ax2.set_zticks(np.linspace(0, 1, 3))
# ax.grid(False)

ax3.plot_wireframe(U, V, M, color='k')
ax3.view_init(elev=15, azim=-135)
# ax3.set_title(r'$M\left(u,v\right) = \min\left(u,v\right)$')
ax3.set_xlabel(r'$F\left(X\right) = u$')
ax3.set_ylabel(r'$G\left(Y\right) = v$')
ax3.zaxis.set_rotate_label(False)  # For being able to rotate Z axis
ax3.set_zlabel(r'$M\left(u,v\right)$', rotation=90)
ax3.set_title(r'$\left(c\right)$', loc='left')
ax3.set_xticks(np.linspace(0, 1, 3))
ax3.set_yticks(np.linspace(0, 1, 3))
ax3.set_zticks(np.linspace(0, 1, 3))
# ax.grid(False)

fig.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------

os.chdir('../../2.0_figures/ch3')
fig.savefig('fig_cop-wmp.pdf')