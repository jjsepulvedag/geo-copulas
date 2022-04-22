# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Graph of histograms and some PDF, Xiaolangdi Hydropower data
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
import sys
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

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
num, cdc, cdp, cuc, cup, uuc, uup = np.loadtxt(fname=file_path, unpack=True)

# Columns 1 and 2 correspond to CD triaxial tests
cdc = cdc[np.logical_not(np.isnan(cdc))]
cdp = cdp[np.logical_not(np.isnan(cdp))]
# Columns 3 and 4 correspond to CU triaxial tests
cuc = cuc[np.logical_not(np.isnan(cuc))]
cup = cup[np.logical_not(np.isnan(cup))]
# Columns 5 and 6 correspond to UU triaxial tests
uuc = uuc[np.logical_not(np.isnan(uuc))]
uup = uup[np.logical_not(np.isnan(uup))]

# %% -------------Define distributions for each test and parameter-------------

# ----------------------Consolidated drained triaxial test---------------------
# ----------------Cohesion----------------
mean = np.mean(cdc)
stnd = np.std(cdc)
cdc_d = {}
cdc_d['truncn'] = ((0 - mean)/stnd, (120 - mean)/stnd, mean, stnd)
cdc_d['logn'] = st.lognorm.fit(cdc, floc=0)
cdc_d['gamma'] = st.gamma.fit(cdc, floc=0)
cdc_d['beta'] = st.beta.fit(cdc, floc=0, fscale=120)
cdc_d['weibull'] = st.weibull_min.fit(cdc, floc=0)
xcdc = np.linspace(0, 120, 15)
ycdc1 = st.truncnorm.pdf(xcdc, a=cdc_d['truncn'][0], b=cdc_d['truncn'][1],
                         loc=cdc_d['truncn'][2], scale=cdc_d['truncn'][3])
ycdc2 = st.lognorm.pdf(xcdc, s=cdc_d['logn'][0], loc=cdc_d['logn'][1],
                       scale=cdc_d['logn'][2])
ycdc3 = st.gamma.pdf(xcdc, a=cdc_d['gamma'][0], loc=cdc_d['gamma'][1],
                     scale=cdc_d['gamma'][2])
ycdc4 = st.beta.pdf(xcdc, a=cdc_d['beta'][0], b=cdc_d['beta'][1],
                    loc=cdc_d['beta'][2], scale=cdc_d['beta'][3])
ycdc5 = st.weibull_min.pdf(xcdc, c=cdc_d['weibull'][0], 
                           loc=cdc_d['weibull'][1], scale=cdc_d['weibull'][2])                    
# -------------Friction angle-------------
mean = np.mean(cdp)
stnd = np.std(cdp)
cdp_d = {}
cdp_d['truncn'] = ((0 - mean)/stnd, (32 - mean)/stnd, mean, stnd)
cdp_d['logn'] = st.lognorm.fit(cdp, floc=0)
cdp_d['gamma'] = st.gamma.fit(cdp, floc=0)
cdp_d['beta'] = st.beta.fit(cdp, floc=0, fscale=32)
cdp_d['weibull'] = st.weibull_min.fit(cdp, floc=0)
xcdp = np.linspace(0, 32, 15)
ycdp1 = st.truncnorm.pdf(xcdp, a=cdp_d['truncn'][0], b=cdp_d['truncn'][1],
                         loc=cdp_d['truncn'][2], scale=cdp_d['truncn'][3])
ycdp2 = st.lognorm.pdf(xcdp, s=cdp_d['logn'][0], loc=cdp_d['logn'][1],
                       scale=cdp_d['logn'][2])
ycdp3 = st.gamma.pdf(xcdp, a=cdp_d['gamma'][0], loc=cdp_d['gamma'][1],
                     scale=cdp_d['gamma'][2])
ycdp4 = st.beta.pdf(xcdp, a=cdp_d['beta'][0], b=cdp_d['beta'][1],
                    loc=cdp_d['beta'][2], scale=cdp_d['beta'][3])
ycdp5 = st.weibull_min.pdf(xcdp, c=cdp_d['weibull'][0], 
                           loc=cdp_d['weibull'][1], scale=cdp_d['weibull'][2])

# ---------------------Consolidated undrained triaxial test--------------------
# ----------------Cohesion----------------
mean = np.mean(cuc)
stnd = np.std(cuc)
cuc_d = {}
cuc_d['truncn'] = ((0 - mean)/stnd, (150 - mean)/stnd, mean, stnd)
cuc_d['logn'] = st.lognorm.fit(cuc, floc=0)
cuc_d['gamma'] = st.gamma.fit(cuc, floc=0)
cuc_d['beta'] = st.beta.fit(cuc, floc=0, fscale=150)
cuc_d['weibull'] = st.weibull_min.fit(cuc, floc=0)
xcuc = np.linspace(0, 150, 15)
ycuc1 = st.truncnorm.pdf(xcuc, a=cuc_d['truncn'][0], b=cuc_d['truncn'][1],
                         loc=cuc_d['truncn'][2], scale=cuc_d['truncn'][3])
ycuc2 = st.lognorm.pdf(xcuc, s=cuc_d['logn'][0], loc=cuc_d['logn'][1],
                       scale=cuc_d['logn'][2])
ycuc3 = st.gamma.pdf(xcuc, a=cuc_d['gamma'][0], loc=cuc_d['gamma'][1],
                     scale=cuc_d['gamma'][2])
ycuc4 = st.beta.pdf(xcuc, a=cuc_d['beta'][0], b=cuc_d['beta'][1],
                    loc=cuc_d['beta'][2], scale=cuc_d['beta'][3])
ycuc5 = st.weibull_min.pdf(xcuc, c=cuc_d['weibull'][0], 
                           loc=cuc_d['weibull'][1], scale=cuc_d['weibull'][2])
# -------------Friction angle-------------
mean = np.mean(cup)
stnd = np.std(cup)
cup_d = {}
cup_d['truncn'] = ((0 - mean)/stnd, (35 - mean)/stnd, mean, stnd)
cup_d['logn'] = st.lognorm.fit(cup, floc=0)
cup_d['gamma'] = st.gamma.fit(cup, floc=0)
cup_d['beta'] = st.beta.fit(cup, floc=0, fscale=32)
cup_d['weibull'] = st.weibull_min.fit(cup, floc=0)
xcup = np.linspace(0, 32, 15)
ycup1 = st.truncnorm.pdf(xcup, a=cup_d['truncn'][0], b=cup_d['truncn'][1],
                         loc=cup_d['truncn'][2], scale=cup_d['truncn'][3])
ycup2 = st.lognorm.pdf(xcup, s=cup_d['logn'][0], loc=cup_d['logn'][1],
                       scale=cup_d['logn'][2])
ycup3 = st.gamma.pdf(xcup, a=cup_d['gamma'][0], loc=cup_d['gamma'][1],
                     scale=cup_d['gamma'][2])
ycup4 = st.beta.pdf(xcup, a=cup_d['beta'][0], b=cup_d['beta'][1],
                    loc=cup_d['beta'][2], scale=cup_d['beta'][3])
ycup5 = st.weibull_min.pdf(xcup, c=cup_d['weibull'][0], 
                           loc=cup_d['weibull'][1], scale=cup_d['weibull'][2])                    


# --------------------Unconsolidated undrained triaxial test-------------------
# ----------------Cohesion----------------
mean = np.mean(uuc)
stnd = np.std(uuc)
uuc_d = {}
uuc_d['truncn'] = ((0 - mean)/stnd, (350 - mean)/stnd, mean, stnd)
uuc_d['logn'] = st.lognorm.fit(uuc, floc=0)
uuc_d['gamma'] = st.gamma.fit(uuc, floc=0)
uuc_d['beta'] = st.beta.fit(uuc, floc=0, fscale=350)
uuc_d['weibull'] = st.weibull_min.fit(uuc, floc=0)
xuuc = np.linspace(0, 350, 15)
yuuc1 = st.truncnorm.pdf(xuuc, a=uuc_d['truncn'][0], b=uuc_d['truncn'][1],
                         loc=uuc_d['truncn'][2], scale=uuc_d['truncn'][3])
yuuc2 = st.lognorm.pdf(xuuc, s=uuc_d['logn'][0], loc=uuc_d['logn'][1],
                       scale=uuc_d['logn'][2])
yuuc3 = st.gamma.pdf(xuuc, a=uuc_d['gamma'][0], loc=uuc_d['gamma'][1],
                     scale=uuc_d['gamma'][2])
yuuc4 = st.beta.pdf(xuuc, a=uuc_d['beta'][0], b=uuc_d['beta'][1],
                    loc=uuc_d['beta'][2], scale=uuc_d['beta'][3])
yuuc5 = st.weibull_min.pdf(xuuc, c=uuc_d['weibull'][0], 
                           loc=uuc_d['weibull'][1], scale=uuc_d['weibull'][2])                     
# -------------Friction angle-------------
mean = np.mean(uup)
stnd = np.std(uup)
uup_d = {}
uup_d['truncn'] = ((0 - mean)/stnd, (25 - mean)/stnd, mean, stnd)
uup_d['logn'] = st.lognorm.fit(uup, floc=0)
uup_d['gamma'] = st.gamma.fit(uup, floc=0)
uup_d['beta'] = st.beta.fit(uup, floc=0, fscale=25)
uup_d['weibull'] = st.weibull_min.fit(uup, floc=0)
xuup = np.linspace(0, 25, 15)
yuup1 = st.truncnorm.pdf(xuup, a=uup_d['truncn'][0], b=uup_d['truncn'][1],
                         loc=uup_d['truncn'][2], scale=uup_d['truncn'][3])
yuup2 = st.lognorm.pdf(xuup, s=uup_d['logn'][0], loc=uup_d['logn'][1],
                       scale=uup_d['logn'][2])
yuup3 = st.gamma.pdf(xuup, a=uup_d['gamma'][0], loc=uup_d['gamma'][1],
                     scale=uup_d['gamma'][2])
yuup4 = st.beta.pdf(xuup, a=uup_d['beta'][0], b=uup_d['beta'][1],
                    loc=uup_d['beta'][2], scale=uup_d['beta'][3])
yuup5 = st.weibull_min.pdf(xuup, c=uup_d['weibull'][0], 
                           loc=uup_d['weibull'][1], scale=uup_d['weibull'][2])


# %% -----------------------Plot the histograms and PDFs-----------------------

fig = plt.figure(figsize=(8.3, 8.3))
bins = 10

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

ax1.hist(cdc, bins=bins, density=True, fill=False, label='Hist')
ax1.plot(xcdc, ycdc1, c='k', marker='.', markersize=5, label='Truncnormal')
ax1.plot(xcdc, ycdc2, c='k', marker='s', markersize=5, label='Lognormal')
ax1.plot(xcdc, ycdc3, c='k', marker='v', markersize=5, label='Gamma')
ax1.plot(xcdc, ycdc4, c='k', marker='d', markersize=5, label='Beta')
ax1.plot(xcdc, ycdc5, c='k', marker='x', markersize=7, label='Weibull')
ax1.set_title(r'$\left(a\right)$', loc='left', fontsize=13)
ax1.set_xlabel(r'$c\left(KPa\right)$')
ax1.set_ylabel(r'$PDF$')
ax1.set_xticks(np.linspace(0, 120, 7))
ax1.set_yticks(np.linspace(0, 0.025, 6))
ax1.set_xlim(0, 120)
ax1.set_ylim(0, 0.025)
ax1.legend(loc=0, prop={'size': 8})

ax2.hist(cdp, bins=bins, density=True, fill=False, label='Hist')
ax2.plot(xcdp, ycdp1, c='k', marker='.', markersize=5, label='Truncnormal')
ax2.plot(xcdp, ycdp2, c='k', marker='s', markersize=5, label='Lognormal')
ax2.plot(xcdp, ycdp3, c='k', marker='v', markersize=5, label='Gamma')
ax2.plot(xcdp, ycdp4, c='k', marker='d', markersize=5, label='Beta')
ax2.plot(xcdp, ycdp5, c='k', marker='x', markersize=7, label='Weibull')
ax2.set_title(r'$\left(b\right)$', loc='left', fontsize=13)
ax2.set_xlabel(r'$\phi\left(^\circ\right)$')
ax2.set_ylabel(r'$PDF$')
ax2.set_xticks(np.linspace(12, 32, 6))
ax2.set_yticks(np.linspace(0, 0.25, 6))
ax2.set_xlim(12, 32)
ax2.set_ylim(0, 0.25)
ax2.legend(loc=0, prop={'size': 8})

ax3.hist(cuc, bins=bins, density=True, fill=False, label='Hist')
ax3.plot(xcuc, ycuc1, c='k', marker='.', markersize=5, label='Truncnormal')
ax3.plot(xcuc, ycuc2, c='k', marker='s', markersize=5, label='Lognormal')
ax3.plot(xcuc, ycuc3, c='k', marker='v', markersize=5, label='Gamma')
ax3.plot(xcuc, ycuc4, c='k', marker='d', markersize=5, label='Beta')
ax3.plot(xcuc, ycuc5, c='k', marker='x', markersize=7, label='Weibull')
ax3.set_title(r'$\left(c\right)$', loc='left', fontsize=13)
ax3.set_xlabel(r'$c\left(KPa\right)$')
ax3.set_ylabel(r'$PDF$')
ax3.set_xticks(np.linspace(0, 150, 7))
ax3.set_yticks(np.linspace(0, 0.02, 6))
ax3.set_xlim(0, 150)
ax3.set_ylim(0, 0.02)
ax3.legend(loc=0, prop={'size': 8})

ax4.hist(cup, bins=bins, density=True, fill=False, label='Hist')
ax4.plot(xcup, ycup1, c='k', marker='.', markersize=5, label='Truncnormal')
ax4.plot(xcup, ycup2, c='k', marker='s', markersize=5, label='Lognormal')
ax4.plot(xcup, ycup3, c='k', marker='v', markersize=5, label='Gamma')
ax4.plot(xcup, ycup4, c='k', marker='d', markersize=5, label='Beta')
ax4.plot(xcup, ycup5, c='k', marker='x', markersize=7, label='Weibull')
ax4.set_title(r'$\left(d\right)$', loc='left', fontsize=13)
ax4.set_xlabel(r'$\phi\left(^\circ\right)$')
ax4.set_ylabel(r'$PDF$')
ax4.set_xticks(np.linspace(12, 32, 6))
ax4.set_yticks(np.linspace(0, 0.25, 6))
ax4.set_xlim(12, 32)
ax4.set_ylim(0, 0.25)
ax4.legend(loc=0, prop={'size': 8})

ax5.hist(uuc, bins=bins, density=True, fill=False, label='Hist')
ax5.plot(xuuc, yuuc1, c='k', marker='.', markersize=5, label='Truncnormal')
ax5.plot(xuuc, yuuc2, c='k', marker='s', markersize=5, label='Lognormal')
ax5.plot(xuuc, yuuc3, c='k', marker='v', markersize=5, label='Gamma')
ax5.plot(xuuc, yuuc4, c='k', marker='d', markersize=5, label='Beta')
ax5.plot(xuuc, yuuc5, c='k', marker='x', markersize=7, label='Weibull')
ax5.set_title(r'$\left(e\right)$', loc='left', fontsize=13)
ax5.set_xlabel(r'$c\left(KPa\right)$')
ax5.set_ylabel(r'$PDF$')
ax5.set_xticks(np.linspace(0, 350, 7))
ax5.set_yticks(np.linspace(0, 0.015, 6))
ax5.set_xlim(0, 350)
ax5.set_ylim(0, 0.015)
ax5.legend(loc=0, prop={'size': 8})

ax6.hist(uup, bins=bins, density=True, fill=False, label='Hist')
ax6.plot(xuup, yuup1, c='k', marker='.', markersize=5, label='Truncnormal')
ax6.plot(xuup, yuup2, c='k', marker='s', markersize=5, label='Lognormal')
ax6.plot(xuup, yuup3, c='k', marker='v', markersize=5, label='Gamma')
ax6.plot(xuup, yuup4, c='k', marker='d', markersize=5, label='Beta')
ax6.plot(xuup, yuup5, c='k', marker='x', markersize=7, label='Weibull')
ax6.set_title(r'$\left(f\right)$', loc='left', fontsize=13)
ax6.set_xlabel(r'$\phi\left(^\circ\right)$')
ax6.set_ylabel(r'$PDF$')
ax6.set_xticks(np.linspace(0, 25, 7))
ax6.set_yticks(np.linspace(0, 0.15, 6))
ax6.set_xlim(0, 25)
ax6.set_ylim(0, 0.15)
ax6.legend(loc=0, prop={'size': 8})

plt.tight_layout()
# plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../../2.0_figures/ch4/ch4.1/')
fig.savefig('fig_histograms_cf.pgf')
