# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering.
K-S test values for the considered marginals, Xiaolangdi Hydropower data
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
Notes: 
1. Considered models: Truncnormal, Lognormal, Gamma, Beta, Weibull
"""

# %% ------------------------Import some needed modules------------------------
import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# %% -------------------------------Import data--------------------------------

# Define the correct path to import data
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

# %% ----------------Fit marginals and compute K-S coefficients----------------

# ----------------------Consolidated drained triaxial test---------------------
# Cohesion
mean = np.mean(cdc)
stnd = np.std(cdc)
cdc_fit = {}
cdc_fit['truncn'] = ((0 - mean)/stnd, (120 - mean)/stnd, mean, stnd)
cdc_fit['logn'] = st.lognorm.fit(cdc, floc=0)
cdc_fit['gamma'] = st.gamma.fit(cdc, floc=0)
cdc_fit['beta'] = st.beta.fit(cdc, floc=0, fscale=120)
cdc_fit['weibull'] = st.weibull_min.fit(cdc, floc=0)
truncn_cdc = lambda x: st.truncnorm.cdf(x, a=cdc_fit['truncn'][0],
                                        b=cdc_fit['truncn'][1],
                                        loc=cdc_fit['truncn'][2],
                                        scale=cdc_fit['truncn'][3])
logn_cdc = lambda x: st.lognorm.cdf(x, s=cdc_fit['logn'][0],
                                    loc=cdc_fit['logn'][1],
                                    scale=cdc_fit['logn'][2])
gamma_cdc = lambda x: st.gamma.cdf(x, a=cdc_fit['gamma'][0],
                                   loc=cdc_fit['gamma'][1],
                                   scale=cdc_fit['gamma'][2])
beta_cdc = lambda x: st.beta.cdf(x, a=cdc_fit['beta'][0], b=cdc_fit['beta'][1],
                                 loc=cdc_fit['beta'][2],
                                 scale=cdc_fit['beta'][3])
weibull_cdc = lambda x: st.weibull_min.cdf(x, c=cdc_fit['weibull'][0],
                                           loc=cdc_fit['weibull'][1],
                                           scale=cdc_fit['weibull'][2])                                                                                          
ks_cdc = np.zeros(6)
func_cdc = (truncn_cdc, logn_cdc, gamma_cdc, beta_cdc, weibull_cdc)
for i in range(len(ks_cdc) - 1):
    ks_cdc[i] = st.kstest(cdc, cdf=func_cdc[i])[0]
Dcdc = 1.36/np.sqrt(len(cdc))
ks_cdc[-1] = Dcdc
# Friction angle
mean = np.mean(cdp)
stnd = np.std(cdp)
cdp_fit = {}
cdp_fit['truncn'] = ((0 - mean)/stnd, (32 - mean)/stnd, mean, stnd)
cdp_fit['logn'] = st.lognorm.fit(cdp, floc=0)
cdp_fit['gamma'] = st.gamma.fit(cdp, floc=0)
cdp_fit['beta'] = st.beta.fit(cdp, floc=0, fscale=32)
cdp_fit['weibull'] = st.weibull_min.fit(cdp, floc=0)
truncn_cdp = lambda x: st.truncnorm.cdf(x, a=cdp_fit['truncn'][0],
                                        b=cdp_fit['truncn'][1], 
                                        loc=cdp_fit['truncn'][2], 
                                        scale=cdp_fit['truncn'][3])
logn_cdp = lambda x: st.lognorm.cdf(x, s=cdp_fit['logn'][0], 
                                    loc=cdp_fit['logn'][1], 
                                    scale=cdp_fit['logn'][2])
gamma_cdp = lambda x: st.gamma.cdf(x, a=cdp_fit['gamma'][0], 
                                   loc=cdp_fit['gamma'][1], 
                                   scale=cdp_fit['gamma'][2]) 
beta_cdp = lambda x: st.beta.cdf(x, a=cdp_fit['beta'][0], b=cdp_fit['beta'][1], 
                                 loc=cdp_fit['beta'][2], 
                                 scale=cdp_fit['beta'][3])
weibull_cdp = lambda x: st.weibull_min.cdf(x, c=cdp_fit['weibull'][0], 
                                           loc=cdp_fit['weibull'][1], 
                                           scale=cdp_fit['weibull'][2])
ks_cdp = np.zeros(6)
func_cdp = (truncn_cdp, logn_cdp, gamma_cdp, beta_cdp, weibull_cdp)
for i in range(len(ks_cdp) - 1):
    ks_cdp[i] = st.kstest(cdp, cdf=func_cdp[i])[0]   
Dcdp = 1.36/np.sqrt(len(cdp))
ks_cdp[-1] = Dcdp

# ---------------------Consolidated undrained triaxial test--------------------
# Cohesion
mean = np.mean(cuc)
stnd = np.std(cuc)
cuc_fit = {}
cuc_fit['truncn'] = ((0 - mean)/stnd, (150 - mean)/stnd, mean, stnd)
cuc_fit['logn'] = st.lognorm.fit(cuc, floc=0)
cuc_fit['gamma'] = st.gamma.fit(cuc, floc=0)
cuc_fit['beta'] = st.beta.fit(cuc, floc=0, fscale=150)
cuc_fit['weibull'] = st.weibull_min.fit(cuc, floc=0)
truncn_cuc = lambda x: st.truncnorm.cdf(x, a=cuc_fit['truncn'][0],
                                        b=cuc_fit['truncn'][1], 
                                        loc=cuc_fit['truncn'][2], 
                                        scale=cuc_fit['truncn'][3])
logn_cuc = lambda x: st.lognorm.cdf(x, s=cuc_fit['logn'][0], 
                                    loc=cuc_fit['logn'][1], 
                                    scale=cuc_fit['logn'][2])
gamma_cuc = lambda x: st.gamma.cdf(x, a=cuc_fit['gamma'][0], 
                                   loc=cuc_fit['gamma'][1], 
                                   scale=cuc_fit['gamma'][2]) 
beta_cuc = lambda x: st.beta.cdf(x, a=cuc_fit['beta'][0], b=cuc_fit['beta'][1], 
                                 loc=cuc_fit['beta'][2], 
                                 scale=cuc_fit['beta'][3])
weibull_cuc = lambda x: st.weibull_min.cdf(x, c=cuc_fit['weibull'][0], 
                                           loc=cuc_fit['weibull'][1], 
                                           scale=cuc_fit['weibull'][2])                                                                                                      
ks_cuc = np.zeros(6)
func_cuc = (truncn_cuc, logn_cuc, gamma_cuc, beta_cuc, weibull_cuc)
for i in range(len(ks_cuc) - 1):
    ks_cuc[i] = st.kstest(cuc, cdf=func_cuc[i])[0] 
Dcuc = 1.36/np.sqrt(len(cuc))
ks_cuc[-1] = Dcuc
# Friction angle
mean = np.mean(cup)
stnd = np.std(cup)
cup_fit = {}
cup_fit['truncn'] = ((0 - mean)/stnd, (32 - mean)/stnd, mean, stnd)
cup_fit['logn'] = st.lognorm.fit(cup, floc=0)
cup_fit['gamma'] = st.gamma.fit(cup, floc=0)
cup_fit['beta'] = st.beta.fit(cup, floc=0, fscale=32)
cup_fit['weibull'] = st.weibull_min.fit(cup, floc=0)
truncn_cup = lambda x: st.truncnorm.cdf(x, a=cup_fit['truncn'][0],
                                        b=cup_fit['truncn'][1], 
                                        loc=cup_fit['truncn'][2], 
                                        scale=cup_fit['truncn'][3])
logn_cup = lambda x: st.lognorm.cdf(x, s=cup_fit['logn'][0], 
                                    loc=cup_fit['logn'][1], 
                                    scale=cup_fit['logn'][2])
gamma_cup = lambda x: st.gamma.cdf(x, a=cup_fit['gamma'][0], 
                                   loc=cup_fit['gamma'][1], 
                                   scale=cup_fit['gamma'][2]) 
beta_cup = lambda x: st.beta.cdf(x, a=cup_fit['beta'][0], b=cup_fit['beta'][1], 
                                 loc=cup_fit['beta'][2], 
                                 scale=cup_fit['beta'][3])
weibull_cup = lambda x: st.weibull_min.cdf(x, c=cup_fit['weibull'][0], 
                                           loc=cup_fit['weibull'][1], 
                                           scale=cup_fit['weibull'][2])
ks_cup = np.zeros(6)
func_cup = (truncn_cup, logn_cup, gamma_cup, beta_cup, weibull_cup)
for i in range(len(ks_cup) - 1):
    ks_cup[i] = st.kstest(cup, cdf=func_cup[i])[0]
Dcup = 1.36/np.sqrt(len(cup))  
ks_cup[-1] = Dcup

# --------------------Unconsolidated undrained triaxial test-------------------
# Cohesion
mean = np.mean(uuc)
stnd = np.std(uuc)
uuc_fit = {}
uuc_fit['truncn'] = ((0 - mean)/stnd, (350 - mean)/stnd, mean, stnd)
uuc_fit['logn'] = st.lognorm.fit(uuc, floc=0)
uuc_fit['gamma'] = st.gamma.fit(uuc, floc=0)
uuc_fit['beta'] = st.beta.fit(uuc, floc=0, fscale=350)
uuc_fit['weibull'] = st.weibull_min.fit(uuc, floc=0)
truncn_uuc = lambda x: st.truncnorm.cdf(x, a=uuc_fit['truncn'][0],
                                        b=uuc_fit['truncn'][1], 
                                        loc=uuc_fit['truncn'][2], 
                                        scale=uuc_fit['truncn'][3])
logn_uuc = lambda x: st.lognorm.cdf(x, s=uuc_fit['logn'][0], 
                                    loc=uuc_fit['logn'][1], 
                                    scale=uuc_fit['logn'][2])
gamma_uuc = lambda x: st.gamma.cdf(x, a=uuc_fit['gamma'][0], 
                                   loc=uuc_fit['gamma'][1], 
                                   scale=uuc_fit['gamma'][2]) 
beta_uuc = lambda x: st.beta.cdf(x, a=uuc_fit['beta'][0], b=uuc_fit['beta'][1], 
                                 loc=uuc_fit['beta'][2], 
                                 scale=uuc_fit['beta'][3])
weibull_uuc = lambda x: st.weibull_min.cdf(x, c=uuc_fit['weibull'][0], 
                                           loc=uuc_fit['weibull'][1], 
                                           scale=uuc_fit['weibull'][2])                                                                                                      
ks_uuc = np.zeros(6)
func_uuc = (truncn_uuc, logn_uuc, gamma_uuc, beta_uuc, weibull_uuc)
for i in range(len(ks_uuc) - 1):
    ks_uuc[i] = st.kstest(uuc, cdf=func_uuc[i])[0] 
Duuc = 1.36/np.sqrt(len(uuc))    
ks_uuc[-1] = Duuc
# Friction angle
mean = np.mean(uup)
stnd = np.std(uup)
uup_fit = {}
uup_fit['truncn'] = ((0 - mean)/stnd, (25 - mean)/stnd, mean, stnd)
uup_fit['logn'] = st.lognorm.fit(uup, floc=0)
uup_fit['gamma'] = st.gamma.fit(uup, floc=0)
uup_fit['beta'] = st.beta.fit(uup, floc=0, fscale=25)
uup_fit['weibull'] = st.weibull_min.fit(uup, floc=0)
truncn_uup = lambda x: st.truncnorm.cdf(x, a=uup_fit['truncn'][0],
                                        b=uup_fit['truncn'][1], 
                                        loc=uup_fit['truncn'][2], 
                                        scale=uup_fit['truncn'][3])
logn_uup = lambda x: st.lognorm.cdf(x, s=uup_fit['logn'][0], 
                                    loc=uup_fit['logn'][1], 
                                    scale=uup_fit['logn'][2])
gamma_uup = lambda x: st.gamma.cdf(x, a=uup_fit['gamma'][0], 
                                   loc=uup_fit['gamma'][1], 
                                   scale=uup_fit['gamma'][2]) 
beta_uup = lambda x: st.beta.cdf(x, a=uup_fit['beta'][0], b=uup_fit['beta'][1], 
                                 loc=uup_fit['beta'][2], 
                                 scale=uup_fit['beta'][3])
weibull_uup = lambda x: st.weibull_min.cdf(x, c=uup_fit['weibull'][0], 
                                           loc=uup_fit['weibull'][1], 
                                           scale=uup_fit['weibull'][2])
ks_uup = np.zeros(6)
func_uup = (truncn_uup, logn_uup, gamma_uup, beta_uup, weibull_uup)
for i in range(len(ks_uup) - 1):
    ks_uup[i] = st.kstest(uup, cdf=func_uup[i])[0]
Duup = 1.36/np.sqrt(len(uup))        
ks_uup[-1] = Duup

# %% ------------------------------Results Table-------------------------------

results_table = np.array([ks_cdc, ks_cdp, ks_cuc, ks_cup, ks_uuc, ks_uup])

np.savetxt('tab_kstests_marginals.txt', results_table,fmt='%.3f')

