# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering.
Table of AIC/BIC values from all the copulas.
Xiaolangdi hydropower station
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Defining the correct path to import pycopulas module and to get data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\..\\')
main_path = os.getcwd()
sys.path.append(main_path)

from pycopulas import gaussian, student, plackett, frank, no16, clayton

# %% -------------------------------Import data--------------------------------

file_path = os.path.join('data', 'xiaolangdi_hydropower.txt')
data = np.loadtxt(fname=file_path, delimiter=' ')

# Columns 1 and 2 correspond to CD triaxial tests
cdc = data[:, 1][np.logical_not(np.isnan(data[:, 1]))]
cdp = data[:, 2][np.logical_not(np.isnan(data[:, 2]))]
# Columns 3 and 4 correspond to CU triaxial tests
cuc = data[:, 3][np.logical_not(np.isnan(data[:, 3]))]
cup = data[:, 4][np.logical_not(np.isnan(data[:, 4]))]
# Columns 5 and 6 correspond to UU triaxial tests
uuc = data[:, 5][np.logical_not(np.isnan(data[:, 5]))]
uup = data[:, 6][np.logical_not(np.isnan(data[:, 6]))]

# %% --------------------Define the empirical distributions--------------------

# Empirical distribution for all the tests
cd_u = st.rankdata(cdc)/(len(cdc) + 1)
cd_v = st.rankdata(cdp)/(len(cdp) + 1)
cu_u = st.rankdata(cuc)/(len(cuc) + 1)
cu_v = st.rankdata(cup)/(len(cup) + 1)
uu_u = st.rankdata(uuc)/(len(uuc) + 1)
uu_v = st.rankdata(uup)/(len(uup) + 1)

emp = np.array([[cd_u, cd_v], [cu_u, cu_v], [uu_u, uu_v]])

# %% ------------------Compute Kendall's tau for each triaxial-----------------

tau_cd = st.kendalltau(cdc, cdp)[0]
tau_cu = st.kendalltau(cuc, cup)[0]
tau_uu = st.kendalltau(uuc, uup)[0]

taus = np.round([tau_cd, tau_cu, tau_uu], 3)

# %% -------------------------Fitting copulas to data--------------------------

thetas = np.zeros((3, 5))

for i in range(len(taus)):
    thetas[i, 0] = gaussian.fit(corr=taus[i], method='moments-t')
    thetas[i, 1] = student.fit(corr=taus[i], method='moments-t')[0]
    thetas[i, 2] = plackett.fit(corr=taus[i], method='moments-t')
    thetas[i, 3] = frank.fit(corr=taus[i], method='moments-t')
    thetas[i, 4] = no16.fit(corr=taus[i], method='moments-t')

thetas = np.round(thetas, 3)
print(thetas[:, 2])

# %% ------------------Compute AIC/BIC values for all copulas------------------    

aics = np.zeros((3, 5))
bics = np.zeros((3, 5))

for i in range(aics.shape[0]):
    aics[i, 0] = gaussian.aic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 0])
    aics[i, 1] = student.aic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 1], nu=4)
    aics[i, 2] = plackett.aic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 2])
    aics[i, 3] = frank.aic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 3])
    aics[i, 4] = no16.aic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 4])

for i in range(bics.shape[0]):
    bics[i, 0] = gaussian.bic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 0])
    bics[i, 1] = student.bic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 1], nu=4)
    bics[i, 2] = plackett.bic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 2])
    bics[i, 3] = frank.bic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 3])
    bics[i, 4] = no16.bic(u=emp[i, 0], v=emp[i, 1], theta=thetas[i, 4])    

# %% ------------------------------Results Table-------------------------------

# Return to the file path
os.chdir(os.path.dirname(os.path.abspath(__file__)))


separator = np.zeros((1, 5))

results_table = np.concatenate((aics, separator, bics))

np.savetxt('tab_aicbic_copulas.txt', results_table, fmt='%.2f')
