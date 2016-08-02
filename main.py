from __future__ import division
import matplotlib
from matplotlib import pylab as pl
import numpy as np
import time


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


def alpha_n(v): return 0.01*(-v+10)/(np.exp((-v+10)/10) - 1) if v!=10 else 0.1
def beta_n(v):  return 0.125*np.exp(-v/80)
def n_inf(v): return alpha_n(-v)/(alpha_n(-v)+beta_n(-v))

def alpha_m(v): return 0.1*(-v+25)/(np.exp((-v+25)/10) - 1 ) if v!=25 else 1
def beta_m(v): return 4*np.exp(-v/18)
def m_inf(v): return alpha_m(-v)/(alpha_m(-v)+beta_m(-v))

def alpha_h(v): return 0.07*np.exp(-v/20)
def beta_h(v): return 1/(np.exp((-v+30)/10)+1)
def h_inf(v): return alpha_h(-v)/(alpha_h(-v)+beta_h(-v))

g_n=120
g_k=36
g_l=0.3
v_n=115
v_k=-12
v_l=10.613
c=1

dt = 20e-2
T = 50.0
time_array = np.arange(0, T, dt)
v, v_euler = np.zeros(len(time_array)), np.zeros(len(time_array))

v_rest = 0
v[0] = v_rest
v_euler[0] = v_rest


I_s = np.zeros(len(time_array))
I_s[:] = 10



def rhs(i_s, m_, n_, h_, v_):
	return (1./c) * (i_s - g_n*(m_)**3*(h_) * (v_ - v_n) - g_k*(n_**4)*(v_ - v_k) - g_l * (v_ - v_l))

def rhs_gating_vars(alpha_, beta_, v_, var_):
    return alpha_(v_) * (1 - var_) - beta_(v_) * var_


step_v = 1e-5
d_gating_var = 1e-5

m = m_inf(v_rest)
h = h_inf(v_rest)
n = n_inf(v_rest)

startEuler = time.clock()
for i in range(1, len(time_array)):

    m += (alpha_m(v_euler[i-1]) * (1-m) - beta_m(v_euler[i-1])*m) * dt
    n += (alpha_n(v_euler[i-1]) * (1-n) - beta_n(v_euler[i-1])*n) * dt
    h += (alpha_h(v_euler[i-1]) * (1-h) - beta_h(v_euler[i-1])*h) * dt

    #print alpha_m(v_euler[i-1]), alpha_n(v_euler[i-1]), alpha_h(v_euler[i-1])
    dv = rhs(I_s[i - 1], m, n, h, v_euler[i - 1]) * dt
    v_euler[i] = v_euler[i - 1] + dv

timeEuler = time.clock() - startEuler


m = m_inf(v_rest)
h = h_inf(v_rest)
n = n_inf(v_rest)

dtStar = dt / 1.0
startMine = time.clock()
for i in range(1, len(time_array)):
        derivative_m = 1.0 / d_gating_var * (rhs_gating_vars(alpha_m, beta_m, v[i-1], m + d_gating_var) -  rhs_gating_vars(alpha_m, beta_m, v[i-1], m))
        derivative_n = 1.0 / d_gating_var * (rhs_gating_vars(alpha_n, beta_n, v[i-1], n + d_gating_var) -  rhs_gating_vars(alpha_n, beta_n, v[i-1], n))
        derivative_h = 1.0 / d_gating_var * (rhs_gating_vars(alpha_h, beta_h, v[i-1], h + d_gating_var) -  rhs_gating_vars(alpha_h, beta_h, v[i-1], h))

        #derivative = (rhs(I_s[i - 1], m, n, h, v[i-1] + 1e-5) - rhs(I_s[i - 1], m, n, h, v[i-1])) / 1e-5

        m += rhs_gating_vars(alpha_m, beta_m, v[i-1], m) * dt / (1 - dtStar * derivative_m)
        n += rhs_gating_vars(alpha_n, beta_n, v[i-1], n) * dt / (1 - dtStar * derivative_n)
        h += rhs_gating_vars(alpha_h, beta_h, v[i-1], h) * dt / (1 - dtStar * derivative_h)

        #rhs = (1./c) * (I_s[i-1] - g_n*m**3*h*(v[i-1]-v_n) - g_k*n**4*(v[i-1]-v_k) - g_l*(v[i-1]-v_l))
        derivative = (rhs(I_s[i - 1], m, n, h, v[i-1] + d_gating_var) - rhs(I_s[i - 1], m, n, h, v[i-1])) / d_gating_var

        dv = rhs(I_s[i - 1], m, n, h, v[i-1]) * dt / (1 - dtStar * derivative)
        v[i] = v[i - 1] + dv
        
        if (i%2 == 0):        
            dtStar = dt / 1.0
        else:
            dtStar = dt

    	#dv = rhs(I_s[i - 1], m, n, h, v_euler[i - 1]) * dt
    	#v_euler[i] = v_euler[i-1] + dv
timeMine = time.clock() - startMine


#fntSize = 20
f, (ax1) = pl.subplots(1, sharex=False, sharey=False)
#ax1.plot(time_array, v_euler, 'b-', label = 'Forward Euler', linewidth = 4)
ax1.plot(time_array, v, 'g-', label = 'Simplified Backward Euler', linewidth = 4)
ax1.set_title('timestep = %.0e ms\n computational time: Forward Euler = %.2e s, Simplified Backward Euler = %.2e s' % (dt, timeEuler, timeMine))
ax1.set_ylabel('action potential, mV')
ax1.legend()
ax1.grid('on')
'''
ax2.plot(time_array, v, 'g', label = 'Simplified Backward Euler', linewidth = 4)
ax2.set_title('timestep = %.2e ms, computational time = %.2e s' % (dt, timeMine))
ax2.legend()
'''
pl.xlabel('time, ms')
pl.ylabel('action potential, mV')
pl.grid('on')
#f.figure()


'''
pl.figure()
pl.plot(time_array, v, 'b-o', linewidth = 3)
pl.title('Forward Euler, dt = %.2e s, computational time = %.2e s' % (dt, timeMine))
pl.legend(('voltage spiking', 'input signal strength'), "upper right")
pl.xlabel("time, s --->")
pl.ylabel("potential difference,V, mV ---->")
pl.grid('on')
pl.figure()
pl.plot(time_array, v_euler, 'g-o', linewidth = 3)
pl.title('Forward Euler, dt = %.2e s, computational time = %.2e s' % (dt, timeEuler))
pl.grid('on')
pl.legend(('voltage spiking','input signal strength'),"upper right")
pl.xlabel("time, s --->")
pl.ylabel("potential difference,V (mV) ---->")
'''
pl.show()
