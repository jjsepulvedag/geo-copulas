# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering.
AIC/BIC values for the considered marginals, Xiaolangdi Hydropower data
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
1. This file must be run from the same directory where it is contained,
otherwise, there may be errors in loading the data from the .txt file.
2. Considered models: Truncnormal, Lognormal, Gamma, Beta
"""

# %% ------------------------Import some needed modules------------------------

# Some standar modules
import os
import numpy as np
import scipy.stats as st

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

# %% --------Fit the distributions and compute the aic/bic coefficients--------

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
cdc_list = np.zeros(10)
cdc_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(cdc, a=cdc_d['truncn'][0],
              b=cdc_d['truncn'][1], loc=cdc_d['truncn'][2],
              scale=cdc_d['truncn'][3])))
cdc_list[1] = 2*np.log(len(cdc)) - 2*np.sum(np.log(st.truncnorm.pdf(cdc,
              a=cdc_d['truncn'][0], b=cdc_d['truncn'][1],
              loc=cdc_d['truncn'][2], scale=cdc_d['truncn'][3])))
cdc_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(cdc, s=cdc_d['logn'][0],
              loc=cdc_d['logn'][1], scale=cdc_d['logn'][2])))
cdc_list[3] = 2*np.log(len(cdc)) - 2*np.sum(np.log(st.lognorm.pdf(cdc,
              s=cdc_d['logn'][0], loc=cdc_d['logn'][1],
              scale=cdc_d['logn'][2])))           
cdc_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(cdc, a=cdc_d['gamma'][0],
              loc=cdc_d['gamma'][1], scale=cdc_d['gamma'][2])))
cdc_list[5] = 2*np.log(len(cdc)) - 2*np.sum(np.log(st.gamma.pdf(cdc,
              a=cdc_d['gamma'][0], loc=cdc_d['gamma'][1],
              scale=cdc_d['gamma'][2])))
cdc_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(cdc, a=cdc_d['beta'][0],
              b=cdc_d['beta'][1], loc=cdc_d['beta'][2],
              scale=cdc_d['beta'][3])))
cdc_list[7] = 2*np.log(len(cdc)) - 2*np.sum(np.log(st.beta.pdf(cdc, 
              a=cdc_d['beta'][0], b=cdc_d['beta'][1], loc=cdc_d['beta'][2], 
              scale=cdc_d['beta'][3])))
cdc_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(cdc, 
              c=cdc_d['weibull'][0], loc=cdc_d['weibull'][1],
              scale=cdc_d['weibull'][2])))
cdc_list[9] = 2*np.log(len(cdc)) - 2*np.sum(np.log(st.weibull_min.pdf(cdc, 
              c=cdc_d['weibull'][0], loc=cdc_d['weibull'][1],
              scale=cdc_d['weibull'][2])))
# Friction angle
mean = np.mean(cdp)
stnd = np.std(cdp)
cdp_d = {}
cdp_d['truncn'] = ((0 - mean)/stnd, (32 - mean)/stnd, mean, stnd)
cdp_d['logn'] = st.lognorm.fit(cdp, floc=0)
cdp_d['gamma'] = st.gamma.fit(cdp, floc=0)
cdp_d['beta'] = st.beta.fit(cdp, floc=0, fscale=32)
cdp_d['weibull'] = st.weibull_min.fit(cdp, floc=0)
cdp_list = np.zeros(10)
cdp_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(cdp, a=cdp_d['truncn'][0], 
              b=cdp_d['truncn'][1], loc=cdp_d['truncn'][2], 
              scale=cdp_d['truncn'][3]))) 
cdp_list[1] = 2*np.log(len(cdp)) - 2*np.sum(np.log(st.truncnorm.pdf(cdp, 
              a=cdp_d['truncn'][0], b=cdp_d['truncn'][1], 
              loc=cdp_d['truncn'][2], scale=cdp_d['truncn'][3])))
cdp_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(cdp, s=cdp_d['logn'][0], 
              loc=cdp_d['logn'][1], scale=cdp_d['logn'][2]))) 
cdp_list[3] = 2*np.log(len(cdp)) - 2*np.sum(np.log(st.lognorm.pdf(cdp, 
              s=cdp_d['logn'][0], loc=cdp_d['logn'][1], 
              scale=cdp_d['logn'][2])))              
cdp_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(cdp, a=cdp_d['gamma'][0], 
              loc=cdp_d['gamma'][1], scale=cdp_d['gamma'][2])))
cdp_list[5] = 2*np.log(len(cdp)) - 2*np.sum(np.log(st.gamma.pdf(cdp, 
              a=cdp_d['gamma'][0], loc=cdp_d['gamma'][1], 
              scale=cdp_d['gamma'][2])))
cdp_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(cdp, a=cdp_d['beta'][0], 
              b=cdp_d['beta'][1], loc=cdp_d['beta'][2], 
              scale=cdp_d['beta'][3])))
cdp_list[7] = 2*np.log(len(cdp)) - 2*np.sum(np.log(st.beta.pdf(cdp, 
              a=cdp_d['beta'][0], b=cdp_d['beta'][1], loc=cdp_d['beta'][2], 
              scale=cdp_d['beta'][3])))
cdp_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(cdp, 
              c=cdp_d['weibull'][0], loc=cdp_d['weibull'][1],
              scale=cdp_d['weibull'][2])))
cdp_list[9] = 2*np.log(len(cdp)) - 2*np.sum(np.log(st.weibull_min.pdf(cdp, 
              c=cdp_d['weibull'][0], loc=cdp_d['weibull'][1],
              scale=cdp_d['weibull'][2])))              

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
cuc_list = np.zeros(10)
cuc_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(cuc, a=cuc_d['truncn'][0],
              b=cuc_d['truncn'][1], loc=cuc_d['truncn'][2],
              scale=cuc_d['truncn'][3])))
cuc_list[1] = 2*np.log(len(cuc)) - 2*np.sum(np.log(st.truncnorm.pdf(cuc,
              a=cuc_d['truncn'][0], b=cuc_d['truncn'][1], 
              loc=cuc_d['truncn'][2], scale=cuc_d['truncn'][3])))
cuc_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(cuc, s=cuc_d['logn'][0],
              loc=cuc_d['logn'][1], scale=cuc_d['logn'][2]))) 
cuc_list[3] = 2*np.log(len(cuc)) - 2*np.sum(np.log(st.lognorm.pdf(cuc,
              s=cuc_d['logn'][0], loc=cuc_d['logn'][1],
              scale=cuc_d['logn'][2])))         
cuc_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(cuc, a=cuc_d['gamma'][0],
              loc=cuc_d['gamma'][1], scale=cuc_d['gamma'][2])))
cuc_list[5] = 2*np.log(len(cuc)) - 2*np.sum(np.log(st.gamma.pdf(cuc,
              a=cuc_d['gamma'][0], loc=cuc_d['gamma'][1],
              scale=cuc_d['gamma'][2])))
cuc_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(cuc, a=cuc_d['beta'][0],
              b=cuc_d['beta'][1], loc=cuc_d['beta'][2],
              scale=cuc_d['beta'][3])))
cuc_list[7] = 2*np.log(len(cuc)) - 2*np.sum(np.log(st.beta.pdf(cuc,
              a=cuc_d['beta'][0], b=cuc_d['beta'][1], loc=cuc_d['beta'][2],
              scale=cuc_d['beta'][3])))
cuc_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(cuc,
              c=cuc_d['weibull'][0], loc=cuc_d['weibull'][1],
              scale=cuc_d['weibull'][2])))
cuc_list[9] = 2*np.log(len(cuc)) - 2*np.sum(np.log(st.weibull_min.pdf(cuc,
              c=cuc_d['weibull'][0], loc=cuc_d['weibull'][1],
              scale=cuc_d['weibull'][2])))               
# Friction angle
mean = np.mean(cup)
stnd = np.std(cup)
cup_d = {}
cup_d['truncn'] = ((0 - mean)/stnd, (35 - mean)/stnd, mean, stnd)
cup_d['logn'] = st.lognorm.fit(cup, floc=0)
cup_d['gamma'] = st.gamma.fit(cup, floc=0)
cup_d['beta'] = st.beta.fit(cup, floc=0, fscale=35)
cup_d['weibull'] = st.weibull_min.fit(cup, floc=0)
cup_list = np.zeros(10)
cup_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(cup, a=cup_d['truncn'][0],
              b=cup_d['truncn'][1], loc=cup_d['truncn'][2],
              scale=cup_d['truncn'][3])))
cup_list[1] = 2*np.log(len(cup)) - 2*np.sum(np.log(st.truncnorm.pdf(cup,
              a=cup_d['truncn'][0], b=cup_d['truncn'][1],
              loc=cup_d['truncn'][2], scale=cup_d['truncn'][3])))
cup_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(cup, s=cup_d['logn'][0],
              loc=cup_d['logn'][1], scale=cup_d['logn'][2])))
cup_list[3] = 2*np.log(len(cup)) - 2*np.sum(np.log(st.lognorm.pdf(cup,
              s=cup_d['logn'][0], loc=cup_d['logn'][1],
              scale=cup_d['logn'][2])))
cup_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(cup, a=cup_d['gamma'][0],
              loc=cup_d['gamma'][1], scale=cup_d['gamma'][2])))
cup_list[5] = 2*np.log(len(cup)) - 2*np.sum(np.log(st.gamma.pdf(cup,
              a=cup_d['gamma'][0], loc=cup_d['gamma'][1],
              scale=cup_d['gamma'][2])))
cup_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(cup, a=cup_d['beta'][0],
              b=cup_d['beta'][1], loc=cup_d['beta'][2],
              scale=cup_d['beta'][3])))
cup_list[7] = 2*np.log(len(cup)) - 2*np.sum(np.log(st.beta.pdf(cup,
              a=cup_d['beta'][0], b=cup_d['beta'][1], loc=cup_d['beta'][2],
              scale=cup_d['beta'][3])))
cup_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(cup,
              c=cup_d['weibull'][0], loc=cup_d['weibull'][1],
              scale=cup_d['weibull'][2])))
cup_list[9] = 2*np.log(len(cup)) - 2*np.sum(np.log(st.weibull_min.pdf(cup,
              c=cup_d['weibull'][0], loc=cup_d['weibull'][1],
              scale=cup_d['weibull'][2])))                

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
uuc_list = np.zeros(10)
uuc_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(uuc, a=uuc_d['truncn'][0], 
              b=uuc_d['truncn'][1], loc=uuc_d['truncn'][2], 
              scale=uuc_d['truncn'][3]))) 
uuc_list[1] = 2*np.log(len(uuc)) - 2*np.sum(np.log(st.truncnorm.pdf(uuc, 
              a=uuc_d['truncn'][0], b=uuc_d['truncn'][1], 
              loc=uuc_d['truncn'][2], scale=uuc_d['truncn'][3])))
uuc_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(uuc, s=uuc_d['logn'][0], 
              loc=uuc_d['logn'][1], scale=uuc_d['logn'][2]))) 
uuc_list[3] = 2*np.log(len(uuc)) - 2*np.sum(np.log(st.lognorm.pdf(uuc, 
              s=uuc_d['logn'][0], loc=uuc_d['logn'][1], 
              scale=uuc_d['logn'][2])))              
uuc_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(uuc, a=uuc_d['gamma'][0], 
              loc=uuc_d['gamma'][1], scale=uuc_d['gamma'][2])))
uuc_list[5] = 2*np.log(len(uuc)) - 2*np.sum(np.log(st.gamma.pdf(uuc, 
              a=uuc_d['gamma'][0], loc=uuc_d['gamma'][1], 
              scale=uuc_d['gamma'][2])))
uuc_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(uuc, a=uuc_d['beta'][0], 
              b=uuc_d['beta'][1], loc=uuc_d['beta'][2], 
              scale=uuc_d['beta'][3])))
uuc_list[7] = 2*np.log(len(uuc)) - 2*np.sum(np.log(st.beta.pdf(uuc, 
              a=uuc_d['beta'][0], b=uuc_d['beta'][1], loc=uuc_d['beta'][2], 
              scale=uuc_d['beta'][3])))
uuc_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(uuc, 
              c=uuc_d['weibull'][0], loc=uuc_d['weibull'][1],
              scale=uuc_d['weibull'][2])))
uuc_list[9] = 2*np.log(len(uuc)) - 2*np.sum(np.log(st.weibull_min.pdf(uuc, 
              c=uuc_d['weibull'][0], loc=uuc_d['weibull'][1],
              scale=uuc_d['weibull'][2])))                 
# Friction angle
mean = np.mean(uup)
stnd = np.std(uup)
uup_d = {}
uup_d['truncn'] = ((0 - mean)/stnd, (25 - mean)/stnd, mean, stnd)
uup_d['logn'] = st.lognorm.fit(uup, floc=0)
uup_d['gamma'] = st.gamma.fit(uup, floc=0)
uup_d['beta'] = st.beta.fit(uup, floc=0, fscale=25)
uup_d['weibull'] = st.weibull_min.fit(uup, floc=0)
uup_list = np.zeros(10)
uup_list[0] = 2*2 - 2*np.sum(np.log(st.truncnorm.pdf(uup, a=uup_d['truncn'][0], 
              b=uup_d['truncn'][1], loc=uup_d['truncn'][2], 
              scale=uup_d['truncn'][3]))) 
uup_list[1] = 2*np.log(len(uup)) - 2*np.sum(np.log(st.truncnorm.pdf(uup, 
              a=uup_d['truncn'][0], b=uup_d['truncn'][1], 
              loc=uup_d['truncn'][2], scale=uup_d['truncn'][3])))
uup_list[2] = 2*2 - 2*np.sum(np.log(st.lognorm.pdf(uup, s=uup_d['logn'][0], 
              loc=uup_d['logn'][1], scale=uup_d['logn'][2]))) 
uup_list[3] = 2*np.log(len(uup)) - 2*np.sum(np.log(st.lognorm.pdf(uup, 
              s=uup_d['logn'][0], loc=uup_d['logn'][1], 
              scale=uup_d['logn'][2])))              
uup_list[4] = 2*2 - 2*np.sum(np.log(st.gamma.pdf(uup, a=uup_d['gamma'][0], 
              loc=uup_d['gamma'][1], scale=uup_d['gamma'][2])))
uup_list[5] = 2*np.log(len(uup)) - 2*np.sum(np.log(st.gamma.pdf(uup, 
              a=uup_d['gamma'][0], loc=uup_d['gamma'][1], 
              scale=uup_d['gamma'][2])))
uup_list[6] = 2*2 - 2*np.sum(np.log(st.beta.pdf(uup, a=uup_d['beta'][0], 
              b=uup_d['beta'][1], loc=uup_d['beta'][2], 
              scale=uup_d['beta'][3])))
uup_list[7] = 2*np.log(len(uup)) - 2*np.sum(np.log(st.beta.pdf(uup, 
              a=uup_d['beta'][0], b=uup_d['beta'][1], loc=uup_d['beta'][2], 
              scale=uup_d['beta'][3])))
uup_list[8] = 2*2 - 2*np.sum(np.log(st.weibull_min.pdf(uup, 
              c=uup_d['weibull'][0], loc=uup_d['weibull'][1],
              scale=uup_d['weibull'][2])))
uup_list[9] = 2*np.log(len(uup)) - 2*np.sum(np.log(st.weibull_min.pdf(uup, 
              c=uup_d['weibull'][0], loc=uup_d['weibull'][1],
              scale=uup_d['weibull'][2])))               


# %% ------------------------------Results Table-------------------------------

results_table = np.array([cdc_list, cdp_list, cuc_list, cup_list, uuc_list, 
                          uup_list], dtype=float)

np.savetxt('tab_aicbic_marginals.txt', results_table,fmt='%.2f')
