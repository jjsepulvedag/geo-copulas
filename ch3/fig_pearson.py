# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of interval of pearson's coefficient with different sigma
-------------------------------------------------------------------------------
Created by    : Juan José Sepúlveda García (jjsepulvedag@unal.edu.co)
Research group: ---
University    : Universidad Nacional de Colombia (unal.edu.co)
-------------------------------------------------------------------------------
First version : April 2020
-------------------------------------------------------------------------------
Based on:
1."Practical approach to dependence modelling using copulas"
   A Dutfoy and R Lebrun.
   2009
   Journal of Risk and Reliability
2."Modelling dependence with copulas and applications to risk management"
   Paul Embrechts et al.
   2001
   Rapport technique, Département de mathématiques, Institut Fédéral de 
   Technologie de Zurich, volume 14.
--------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

# %% ----------------------------- MAIN CODE ----------------------------------

sigma = np.linspace(0.1, 5, 100) 

p_min = ((np.exp(-sigma)-1) /
         (np.sqrt(np.exp(1)-1) * np.sqrt(np.exp(sigma**2)-1)))
p_max = ((np.exp(sigma)-1) /
         (np.sqrt(np.exp(1)-1) * np.sqrt(np.exp(sigma**2)-1)))

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111)

ax1.plot(sigma, p_min, color='k', linestyle='-', label=r'$\rho_{min}$')
ax1.plot(sigma, p_max, color='k', linestyle='-.', label=r'$\rho_{max}$')
ax1.plot(sigma, np.zeros(100), color='k', linestyle='dotted')
ax1.set_xlabel(r'$\sigma$')
ax1.set_ylabel(r'$\rho$')
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 5)
ax1.legend()
ax1.grid()
fig.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------

os.chdir('../../2.0_figures/ch3')
fig.savefig('fig_pearson.pgf')
