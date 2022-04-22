# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Global dispersion factors associated with Probabilities of failure of an 
infinite slope, using different copulas and varying some parameters
--------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
--------------------------------------------------------------------------
First version : July 2020
--------------------------------------------------------------------------
Based on:
1."Impact of copula selection on geotechnical reliability under incomplete
probability information"
Tang X-S et al
2013
Computers and geotechnics
Elsevier
2."Copula-based approaches for evaluating slope reliability under
incomplete probability information"
Tang X-S et al
2015
Structural Safety
Elsevier
--------------------------------------------------------------------------
"""

# %% -----------------------Importing some needed modules----------------------
# Some standar modules
import os
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

initial_time = time.time()

# Defining the correct path to import data
os.chdir(os.path.dirname(os.path.abspath(__file__)))

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# %% ---------------------Import all the tables of results---------------------
# For all the tables, row1: FSn, row2:pf Gauss cop, row3:pf Stud cop,
# row4:pf Plack cop, row5:pf Frank cop, row6:pf No16 cop

file_path_height = os.path.join('result_tables', 'height_resultstable.txt')
file_path_alpha = os.path.join('result_tables', 'alpha_resultstable.txt')
file_path_lambda = os.path.join('result_tables', 'lambda_resultstable.txt')
file_path_tau = os.path.join('result_tables', 'tau_resultstable.txt')

results_table_height = np.loadtxt(fname=file_path_height)
results_table_alpha = np.loadtxt(fname=file_path_alpha)
results_table_lambda = np.loadtxt(fname=file_path_lambda)
results_table_tau = np.loadtxt(fname=file_path_tau)

# %% -------------------Define the curves of each condition--------------------

x_height = results_table_height[0, :]
y_height = np.zeros(len(x_height))
for i in range(results_table_height.shape[1]):
    max_pf = np.max(results_table_height[1:, i])
    min_pf = np.min(results_table_height[1:, i])
    y_height[i] = max_pf/min_pf

x_alpha = results_table_alpha[0, :]
y_alpha = np.zeros(len(x_alpha))
for i in range(results_table_alpha.shape[1]):
    max_pf = np.max(results_table_alpha[1:, i])
    min_pf = np.min(results_table_alpha[1:, i])
    y_alpha[i] = max_pf/min_pf

x_lambda = results_table_lambda[0, :]
y_lambda = np.zeros(len(x_lambda))
for i in range(results_table_lambda.shape[1]):
    max_pf = np.max(results_table_lambda[1:, i])
    min_pf = np.min(results_table_lambda[1:, i])
    y_lambda[i] = max_pf/min_pf

x_tau = results_table_height[0, :]
y_tau = np.zeros(len(x_tau))
for i in range(results_table_tau.shape[1]):
    max_pf = np.max(results_table_tau[1:, i])
    min_pf = np.min(results_table_tau[1:, i])
    y_tau[i] = max_pf/min_pf

# %% ------------------------------Plotting area-------------------------------

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

axes = [ax1, ax2, ax3, ax4]
axes_x = [x_height, x_alpha, x_lambda, x_tau]
axes_y = [y_height, y_alpha, y_lambda, y_tau]

titles = [r'a', r'b', r'c', r'd']
x_lim = np.array([1, 1.16])
y_lim = np.array([[10**0, 10**2],
                  [10**0, 10**2],
                  [10**0, 10**3],
                  [10**0, 10**4]])

x = np.linspace(1, 1.16, 5)
r_1 = np.ones(len(x))*1.5
r_2 = np.ones(len(x))*10

for i in range(len(axes)):
    axes[i].plot(axes_x[i], axes_y[i], marker='s', markersize=3, color='k',
                 label=r'Computed $r$')
    axes[i].plot(x, r_1, linestyle=(0, (1, 1)), color='k', label=r'$r = 1.5$')
    axes[i].plot(x, r_2, linestyle='dashdot', color='k', label=r'$r = 10$')
    axes[i].set_yscale('log')
    axes[i].set_xlabel(r'$FS_n$')

    axes[i].set_xlim(x_lim[0], x_lim[1])
    axes[i].set_ylim(y_lim[i, 0], y_lim[i, 1])
    axes[i].set_xticks(np.linspace(1, 1.16, 5))
    axes[i].set_title('({})'.format(titles[i]), loc='left', fontsize=13)
    axes[i].legend()

plt.tight_layout()
plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../2.0_figures/ch5/')
fig.savefig('infslope_globdispfact.pdf')