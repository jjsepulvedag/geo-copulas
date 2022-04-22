# -*- coding: utf-8 -*-
r"""
--------------------------------------------------------------------------
Article: On the use of copulas in geotechnical engineering
Probabilities of failure of an infinite slope, using different copulas
and varying some parameters
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
2."Risk and reliability in geotechnical engineering"
Phoon KK and Ching J
2014
CRC press
--------------------------------------------------------------------------
"""

# %% -----------------------Importing some needed modules----------------------
# Some standar modules
import os
import sys
import time
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

# Defining the correct path to import pycopulas module and to get data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..\\')
main_path = os.getcwd()
sys.path.append(main_path)

from pycopulas import gaussian, student, plackett, frank, no16

# The following lines are needed to save the final figure in the correc format
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

initial_time = time.time()

# %% -----------Main function to compute the FS of the infinite slope----------


def fs_slope(c, phi, gamma, h, alpha):
    """ Infinite slope formula for the FS """
    num = c + gamma*h*(np.cos(alpha)**2)*np.tan(phi)
    den = gamma*h*np.sin(alpha)*np.cos(alpha)

    return num/den


def fit_lognorm(mean, std):
    """ Adjust a lognormal distribution from mean and std of a Normal dist """
    p = 2*np.log(mean) - 0.5*np.log(std**2 + mean**2)
    q = np.sqrt(np.log(std**2 + mean**2) - np.log(mean**2))

    return p, q


# %% ----------Defining some important values from the slope analysis----------

gamma = 17  # kN/m3
H = 5  # m
alpha = 30  # °

mean_c = 12  # kPa
cv_c = 0.4
mean_p = 30  # kPa
cv_p = 0.2

tau = - 0.5

# %% ----------------------Adjust LogNormal distributions----------------------

p_c, q_c = fit_lognorm(mean_c, cv_c*mean_c)
p_p, q_p = fit_lognorm(mean_p, cv_p*mean_p)

dist_c = st.lognorm(s=q_c, loc=0, scale=np.exp(p_c))
dist_p = st.lognorm(s=q_p, loc=0, scale=np.exp(p_p))

# %% -------------Defining some important values to the simulation-------------

n = 15  # Number of Probabilities of failure calculated per each copula
size = 1_000_000  # Sample size

# Fit each copula to the given Kendall's Tau
theta_g = gaussian.fit(corr=tau, method='moments-t')
theta_s, nu_s = student.fit(corr=tau, method='moments-t')
theta_p = plackett.fit(corr=tau, method='moments-t')
theta_f = frank.fit(corr=tau, method='moments-t')
theta_n = no16.fit(corr=tau, method='moments-t')

# %% ------------------Probabilities of failure by varying H-------------------

h = np.linspace(6.27, 3.84, n)  # Slope Heights to be evaluated

# Vectors where the Pf and norminal FS are going to be stored
fsn_h = np.zeros((1, n))  # This vector will contain the nominal FoS (axis x)
pof_h = np.zeros((5, n))  # r0:Gauss, r1:stud r2:Plack, r3:Frank, r4:No16

# Random sampling from the different copulas
u_g, v_g = gaussian.rvs(p=theta_g, n=size)
u_s, v_s = student.rvs(theta=theta_s, nu=nu_s, n=size)
u_p, v_p = plackett.rvs(theta=theta_p, n=size)
u_f, v_f = frank.rvs(theta=theta_f, n=size)
u_n, v_n = no16.rvs(theta=theta_n, n=size)

c_g, p_g = dist_c.ppf(u_g), dist_p.ppf(v_g)  # c:cohesion, p:friction angle
c_s, p_s = dist_c.ppf(u_s), dist_p.ppf(v_s)
c_p, p_p = dist_c.ppf(u_p), dist_p.ppf(v_p)
c_f, p_f = dist_c.ppf(u_f), dist_p.ppf(v_f)
c_n, p_n = dist_c.ppf(u_n), dist_p.ppf(v_n)

for i in range(n):
    fs_g = fs_slope(c_g, np.radians(p_g), gamma, h[i], np.radians(alpha))
    pof_h[0, i] = sum(j < 1 for j in fs_g)/len(fs_g)

    fs_s = fs_slope(c_s, np.radians(p_s), gamma, h[i], np.radians(alpha))
    pof_h[1, i] = sum(j < 1 for j in fs_s)/len(fs_s)

    fs_p = fs_slope(c_p, np.radians(p_p), gamma, h[i], np.radians(alpha))
    pof_h[2, i] = sum(j < 1 for j in fs_p)/len(fs_p)

    fs_f = fs_slope(c_f, np.radians(p_f), gamma, h[i], np.radians(alpha))
    pof_h[3, i] = sum(j < 1 for j in fs_f)/len(fs_f)

    fs_n = fs_slope(c_n, np.radians(p_n), gamma, h[i], np.radians(alpha))
    pof_h[4, i] = sum(j < 1 for j in fs_n)/len(fs_n)

    fsn_h[0, i] = np.quantile(fs_p, 0.05)

print(fsn_h)

# %% ----------------Probabilities of failure by varying alpha-----------------

alphas = np.linspace(32, 27.68, n)

# Defining vectors where the PoF of each copula are going to be stored
fsn_a = np.zeros((1, n))  # This vector will contain the nominal FoS (axis x)
pof_a = np.zeros((5, n))

# Random sampling from the different copulas
u_g, v_g = gaussian.rvs(p=theta_g, n=size)
u_s, v_s = student.rvs(theta=theta_s, nu=nu_s, n=size)
u_p, v_p = plackett.rvs(theta=theta_p, n=size)
u_f, v_f = frank.rvs(theta=theta_f, n=size)
u_n, v_n = no16.rvs(theta=theta_n, n=size)

c_g, p_g = dist_c.ppf(u_g), dist_p.ppf(v_g)  # c: cohesion, p: friction angle
c_s, p_s = dist_c.ppf(u_s), dist_p.ppf(v_s)
c_p, p_p = dist_c.ppf(u_p), dist_p.ppf(v_p)
c_f, p_f = dist_c.ppf(u_f), dist_p.ppf(v_f)
c_n, p_n = dist_c.ppf(u_n), dist_p.ppf(v_n)

for i in range(n):
    fs_g = fs_slope(c_g, np.radians(p_g), gamma, H, np.radians(alphas[i]))
    pof_a[0, i] = sum(j < 1 for j in fs_g)/len(fs_g)

    fs_s = fs_slope(c_s, np.radians(p_s), gamma, H, np.radians(alphas[i]))
    pof_a[1, i] = sum(j < 1 for j in fs_s)/len(fs_s)

    fs_p = fs_slope(c_p, np.radians(p_p), gamma, H, np.radians(alphas[i]))
    pof_a[2, i] = sum(j < 1 for j in fs_p)/len(fs_p)

    fs_f = fs_slope(c_f, np.radians(p_f), gamma, H, np.radians(alphas[i]))
    pof_a[3, i] = sum(j < 1 for j in fs_f)/len(fs_f)

    fs_n = fs_slope(c_n, np.radians(p_n), gamma, H, np.radians(alphas[i]))
    pof_a[4, i] = sum(j < 1 for j in fs_n)/len(fs_n)

    fsn_a[0, i] = np.quantile(fs_p, 0.05)

print(fsn_a)

# %% -----------------Probabilities of failure by varying cov------------------

lambda_cp = np.linspace(0.77, 1.637, n)

# Defining vectors where the PoF of each copula are going to be stored
fsn_l = np.zeros((1, n))  # This vector will contain the nominal FoS (axis x)
pof_l = np.zeros((5, n))

# Random sampling from the different copulas
u_g, v_g = gaussian.rvs(p=theta_g, n=size)
u_s, v_s = student.rvs(theta=theta_s, nu=nu_s, n=size)
u_p, v_p = plackett.rvs(theta=theta_p, n=size)
u_f, v_f = frank.rvs(theta=theta_f, n=size)
u_n, v_n = no16.rvs(theta=theta_n, n=size)

for i in range(n):
    p_c_temp, q_c_temp = fit_lognorm(mean_c, cv_c*mean_c/lambda_cp[i])
    p_p_temp, q_p_temp = fit_lognorm(mean_p, cv_p*mean_p/lambda_cp[i])

    dist_c_temp = st.lognorm(s=q_c_temp, loc=0, scale=np.exp(p_c_temp))
    dist_p_temp = st.lognorm(s=q_p_temp, loc=0, scale=np.exp(p_p_temp))

    c_g, p_g = dist_c_temp.ppf(u_g), dist_p_temp.ppf(v_g)
    c_s, p_s = dist_c_temp.ppf(u_s), dist_p_temp.ppf(v_s)
    c_p, p_p = dist_c_temp.ppf(u_p), dist_p_temp.ppf(v_p)
    c_f, p_f = dist_c_temp.ppf(u_f), dist_p_temp.ppf(v_f)
    c_n, p_n = dist_c_temp.ppf(u_n), dist_p_temp.ppf(v_n)

    fs_g = fs_slope(c_g, np.radians(p_g), gamma, H, np.radians(alpha))
    pof_l[0, i] = sum(j < 1 for j in fs_g)/len(fs_g)

    fs_s = fs_slope(c_s, np.radians(p_s), gamma, H, np.radians(alpha))
    pof_l[1, i] = sum(j < 1 for j in fs_s)/len(fs_s)

    fs_p = fs_slope(c_p, np.radians(p_p), gamma, H, np.radians(alpha))
    pof_l[2, i] = sum(j < 1 for j in fs_p)/len(fs_p)

    fs_f = fs_slope(c_f, np.radians(p_f), gamma, H, np.radians(alpha))
    pof_l[3, i] = sum(j < 1 for j in fs_f)/len(fs_f)

    fs_n = fs_slope(c_n, np.radians(p_n), gamma, H, np.radians(alpha))
    pof_l[4, i] = sum(j < 1 for j in fs_n)/len(fs_n)

    fsn_l[0, i] = np.quantile(fs_p, 0.05)

print(fsn_l)


# %% -----------------Probabilities of failure by varying tau------------------

taus = np.linspace(-0.26, -0.77, n)

# Defining vectors where the PoF of each copula are going to be stored
fsn_t = np.zeros((1, n))  # This vector will contain the nominal FoS (axis x)
pof_t = np.zeros((5, n))


for i in range(n):

    # Fit each copula to the given Kendall's Tau
    theta_g_temp = gaussian.fit(corr=taus[i], method='moments-t')
    theta_s_temp, nu_s_temp = student.fit(corr=taus[i], method='moments-t') 
    theta_p_temp = plackett.fit(corr=taus[i], method='moments-t')
    theta_f_temp = frank.fit(corr=taus[i], method='moments-t')
    theta_n_temp = no16.fit(corr=taus[i], method='moments-t')

    # Random sampling from the different copulas
    u_g, v_g = gaussian.rvs(p=theta_g_temp, n=size)
    u_s, v_s = student.rvs(theta=theta_s_temp, nu=nu_s_temp, n=size)
    u_p, v_p = plackett.rvs(theta=theta_p_temp, n=size)
    u_f, v_f = frank.rvs(theta=theta_f_temp, n=size)
    u_n, v_n = no16.rvs(theta=theta_n_temp, n=size)

    c_g, p_g = dist_c.ppf(u_g), dist_p.ppf(v_g)
    c_s, p_s = dist_c.ppf(u_s), dist_p.ppf(v_s)
    c_p, p_p = dist_c.ppf(u_p), dist_p.ppf(v_p)
    c_f, p_f = dist_c.ppf(u_f), dist_p.ppf(v_f)
    c_n, p_n = dist_c.ppf(u_n), dist_p.ppf(v_n)

    fs_g = fs_slope(c_g, np.radians(p_g), gamma, H, np.radians(alpha))
    pof_t[0, i] = sum(j < 1 for j in fs_g)/len(fs_g)

    fs_s = fs_slope(c_s, np.radians(p_s), gamma, H, np.radians(alpha))
    pof_t[1, i] = sum(j < 1 for j in fs_s)/len(fs_s)

    fs_p = fs_slope(c_p, np.radians(p_p), gamma, H, np.radians(alpha))
    pof_t[2, i] = sum(j < 1 for j in fs_p)/len(fs_p)

    fs_f = fs_slope(c_f, np.radians(p_f), gamma, H, np.radians(alpha))
    pof_t[3, i] = sum(j < 1 for j in fs_f)/len(fs_f)

    fs_n = fs_slope(c_n, np.radians(p_n), gamma, H, np.radians(alpha))
    pof_t[4, i] = sum(j < 1 for j in fs_n)/len(fs_n)

    fsn_t[0, i] = np.quantile(fs_p, 0.05)

print(fsn_t)

# %% -----------------Export a result table for each condition-----------------

os.chdir(os.path.dirname(os.path.abspath(__file__)))

height_resultstable = np.concatenate((fsn_h, pof_h), axis=0)
np.savetxt('height_resultstable.txt', height_resultstable)

alpha_resultstable = np.concatenate((fsn_a, pof_a), axis=0)
np.savetxt('alpha_resultstable.txt', alpha_resultstable)

lambda_resultstable = np.concatenate((fsn_l, pof_l), axis=0)
np.savetxt('lambda_resultstable.txt', lambda_resultstable)

tau_resultstable = np.concatenate((fsn_t, pof_t), axis=0)
np.savetxt('tau_resultstable.txt', tau_resultstable)


# %% ------------------------------Plotting area-------------------------------

fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

axes = [ax1, ax2, ax3, ax4]
fs = np.array([fsn_h[0, :], fsn_a[0, :], fsn_l[0, :], fsn_t[0, :]])
pf = np.array([pof_h, pof_a, pof_l, pof_t])
titles = [r'(a)', r'(b)', r'(c)', r'(d)']
lgn = [r'Gaussian copula', r'Student-t copula', r'Plackett copula',
       r'Frank copula', r'No. 16 copula']
mark = ['.', 's', 'v', 'd', '.']
x_lim = np.array([1, 1.16])
y_lim = np.array([[10**-4, 10**-1],
                  [10**-4, 10**-1],
                  [10**-5, 10**-1],
                  [10**-6, 10**-1]])
# marker=mark[j], markersize=5
for i in range(len(axes)):
    for j in range(len(lgn)):
        axes[i].plot(fs[i], pf[i, j, :], label=lgn[j])
    axes[i].legend(fontsize='small')
    axes[i].set_yscale('log')
    axes[i].set_xlim(x_lim[0], x_lim[1])
    axes[i].set_ylim(y_lim[i, 0], y_lim[i, 1])
    axes[i].set_xticks(np.linspace(1, 1.16, 5))
    axes[i].set_title('{}'.format(titles[i]), loc='left', fontsize=13)
    axes[i].set_xlabel(r'$FS_n$')
    axes[i].set_ylabel(r'Probability of failure')

final_time = time.time()
print('elapsed time: {}'.format(final_time - initial_time))
plt.tight_layout()
plt.show()

# %%------------------Save the figure in the correct folder--------------------
# The following step is needed to save the final figure in the correct folder
os.chdir(os.path.dirname(__file__))

os.chdir('../../2.0_figures/ch5/')
fig.savefig('pf_infiniteslope.pdf')