# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of dependence c-f for different triaxial test Xiaolangdi Hydropower
--------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
--------------------------------------------------------------------------
First version : June 2020
--------------------------------------------------------------------------
Based on:
1."Bivariate distribution of shear strength parameters using copulas and 
   its impact on geotechnical system reliability"
   Li DQ et al.
   2015
   Elsevier
--------------------------------------------------------------------------
"""

# %% ------------------------Import some needed modules------------------------

# Some standar modules
import os
import numpy as np 
import scipy.stats as st 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# %% -------------------------------Import data--------------------------------

# Defining the correct path to import data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\..\\')

file_path = os.path.join('data', 'xiaolangdi_hydropower.txt')
data = np.loadtxt(fname=file_path, delimiter=' ')

# Columns 1 and 2 correspond to CD triaxial tests
cd_c = data[:, 1][np.logical_not(np.isnan(data[:, 1]))]
cd_p = data[:, 2][np.logical_not(np.isnan(data[:, 2]))]
# Columns 3 and 4 correspond to CU triaxial tests
cu_c = data[:, 3][np.logical_not(np.isnan(data[:, 3]))]
cu_p = data[:, 4][np.logical_not(np.isnan(data[:, 4]))]
# Columns 5 and 6 correspond to UU triaxial tests
uu_c = data[:, 5][np.logical_not(np.isnan(data[:, 5]))]
uu_p = data[:, 6][np.logical_not(np.isnan(data[:, 6]))]

# %% --------------------Define the empirical distributions--------------------

# Empirical distribution for all the tests 
cd_u = st.rankdata(cd_c)/(len(cd_c) + 1)
cd_v = st.rankdata(cd_p)/(len(cd_p) + 1)
cu_u = st.rankdata(cu_c)/(len(cu_c) + 1)
cu_v = st.rankdata(cu_p)/(len(cu_p) + 1)
uu_u = st.rankdata(uu_c)/(len(uu_c) + 1)
uu_v = st.rankdata(uu_p)/(len(uu_p) + 1)

# %% ------------------Plot the graphs for each triaxial test------------------

plt.figure(figsize=(4.5, 4.5))
gs = gs.GridSpec(nrows=2, ncols=4)
ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1, 1:3])

ticks = np.linspace(0, 1, 5)
line_x = np.linspace(0, 1, 3)
line_y = np.linspace(1, 0, 3)
size = 5  # size of the markers in the scatter plots

axes = [ax1, ax2, ax3]
u_list = [cd_u, cu_u, uu_u]
v_list = [cd_v, cu_v, uu_v]
titles = ['a', 'b', 'c']

for i in range(len(axes)):
    axes[i].scatter(u_list[i], v_list[i], s=size, color='k')
    axes[i].plot(line_x, line_y, linewidth=1, linestyle='--', color='k')
    axes[i].set_title('({})'.format(titles[i]), loc='left', fontsize=13)
    axes[i].set_xlabel(r'$u_c$')
    axes[i].set_ylabel(r'$u_{\phi}$')
    axes[i].set_xticks(ticks)
    axes[i].set_yticks(ticks)
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)
    axes[i].grid(linestyle='--')


plt.tight_layout()  # To separate the graphs
plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.1')
plt.savefig('fig_neg_depen_cf.pdf')


# # %% ------------------Plot the graphs for each triaxial test------------------

# plt.figure(figsize=(6, 3))
# gs = gs.GridSpec(nrows=1, ncols=2)
# ax1 = plt.subplot(gs[0, :1])
# ax2 = plt.subplot(gs[0, 1:])

# ticks = np.linspace(0, 1, 5)
# line_x = np.linspace(0, 1, 3)
# line_y = np.linspace(1, 0, 3)
# size = 5  # size of the markers in the scatter plots

# axes = [ax1, ax2]
# u_list = [cd_u, cu_u]
# v_list = [cd_v, cu_v]
# titles = ['a', 'b', 'c']

# for i in range(len(axes)):
#     axes[i].scatter(u_list[i], v_list[i], s=size, color='k')
#     axes[i].plot(line_x, line_y, linewidth=1, linestyle='--', color='k')
#     axes[i].set_title('({})'.format(titles[i]), loc='left', fontsize=12)
#     axes[i].set_xlabel(r'$u_c$')
#     axes[i].set_ylabel(r'$u_{\phi}$')
#     axes[i].set_xticks(ticks)
#     axes[i].set_yticks(ticks)
#     axes[i].set_xlim(0, 1)
#     axes[i].set_ylim(0, 1)
#     axes[i].grid(linestyle='--')


# plt.tight_layout()  # To separate the graphs
# # plt.show()

# # %%------------------Save the figure in the correct folder--------------------
# # The following step is needed to save the final figure in the correct folder
# os.chdir(os.path.dirname(__file__))

# os.chdir('../../../2.0_figures/ch4/ch4.1')
# plt.savefig('fig_neg_depen_cf.pdf')