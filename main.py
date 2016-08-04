from __future__ import division
import matplotlib
from matplotlib import pylab as pl
import numpy as np
import time

font = {'family': 'normal',
        'weight': 'normal',
        'size': 20}

matplotlib.rc('font', **font)


def alpha_n(v): return 0.01 * (-v + 10) / (np.exp((-v + 10) / 10) - 1) if v != 10 else 0.1


def beta_n(v):  return 0.125 * np.exp(-v / 80)


def n_inf(v): return alpha_n(v) / (alpha_n(v) + beta_n(v))


def alpha_m(v): return 0.1 * (-v + 25) / (np.exp((-v + 25) / 10) - 1) if v != 25 else 1


def beta_m(v): return 4 * np.exp(-v / 18)


def m_inf(v): return alpha_m(v) / (alpha_m(v) + beta_m(v))


def alpha_h(v): return 0.07 * np.exp(-v / 20)


def beta_h(v): return 1 / (np.exp((-v + 30) / 10) + 1)


def h_inf(v): return alpha_h(v) / (alpha_h(v) + beta_h(v))


def CalculateNorm(arrayAnalitical, array2):
    k = len(arrayAnalitical) / len(array2)
    normArray = []
    for i in range(len(array2)):
        delta = abs(arrayAnalitical[int(k * i)] - array2[i])
        normArray.append(delta)
    normArray = np.array(normArray)
    max = np.amax(normArray)
    indexMax = np.argmax(normArray)
    return max, indexMax





g_n = 120
g_k = 36
g_l = 0.3
v_n = 115
v_k = -12
v_l = 10.613
c = 1

dtArray = np.linspace(1e-4, 5e-2, 10)
#dtArray.insert(0, 1e-4)
print dtArray
T = 5.0


def rhs(i_s, m_, n_, h_, v_):
    return (1. / c) * (i_s - g_n * (m_) ** 3 * (h_) * (v_ - v_n) - g_k * (n_ ** 4) * (v_ - v_k) - g_l * (v_ - v_l))


def rhs_gating_vars(alpha_, beta_, v_, var_):
    return alpha_(v_) * (1 - var_) - beta_(v_) * var_


step_v = 1e-5
d_gating_var = 1e-7
v_rest = 0

'''
startEuler = time.clock()
for i in range(1, len(time_array)):

    m += (alpha_m(v_euler[i-1]) * (1-m) - beta_m(v_euler[i-1])*m) * dt
    n += (alpha_n(v_euler[i-1]) * (1-n) - beta_n(v_euler[i-1])*n) * dt
    h += (alpha_h(v_euler[i-1]) * (1-h) - beta_h(v_euler[i-1])*h) * dt

    #print alpha_m(v_euler[i-1]), alpha_n(v_euler[i-1]), alpha_h(v_euler[i-1])
    dv = rhs(I_s[i - 1], m, n, h, v_euler[i - 1]) * dt
    v_euler[i] = v_euler[i - 1] + dv

timeEuler = time.clock() - startEuler
'''

error, timeMaxError = [], []

counter = 0
pl.figure()
for dt in dtArray:
    startMine = time.clock()

    # grid arrays and parameters
    time_array = np.arange(0, T, dt)
    v, v_euler, m, n, h, Rhs = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))

    if counter == 0:
        analiticalSolution = np.zeros(len(time_array))



    m[0] = m_inf(v_rest)
    h[0] = h_inf(v_rest)
    n[0] = n_inf(v_rest)
    v[0] = v_rest

    I_s = np.zeros(len(time_array))
    I_s[:] = 10

    dtStar = dt / 2
    for i in range(1, len(time_array)):

        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))

        # rhs = (1./c) * (I_s[i-1] - g_n*m**3*h*(v[i-1]-v_n) - g_k*n**4*(v[i-1]-v_k) - g_l*(v[i-1]-v_l))
        derivative = (rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1] + d_gating_var) - rhs(I_s[i - 1], m[i], n[i], h[i],
                                                                              v[i - 1])) / d_gating_var

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1])

        if counter != 0: # for calculating numerical solutions with various timesteps
            dv = Rhs[i-1] * dt# / (1 - dtStar * derivative)
        else: # for calculating "analitical" solution using Explicit Euler
            dv = Rhs[i-1] * dt

        v[i] = v[i - 1] + dv

        '''if (i % 2 == 0):
            dtStar = dt / 1
        else:
            dtStar = dt'''

    if (counter == 0):
        analiticalSolution[:] = v[:]
        #print 'ana = ', analiticalSolution


            # dv = rhs(I_s[i - 1], m, n, h, v_euler[i - 1]) * dt
            # v_euler[i] = v_euler[i-1] + dv
    Rhs[-1] = Rhs[-2]
    timeMine = time.clock() - startMine

    errTmp, indexErrTpm = CalculateNorm(analiticalSolution, v)
    error.append(errTmp)
    timeMaxError.append(indexErrTpm * dt)

    if counter == 0:
        pl.plot(time_array, v, '--k', label='analytical (dt = %.2e ms)' % dtArray[counter], linewidth=5)
    elif (counter%1 == 0):
        pl.plot(time_array, v, '-', label='dt = %.2e ms' % dtArray[counter], linewidth=3)
    #pl.title('timestep = %.0e ms\n' % (dt))
    pl.legend(loc='upper left')
    pl.xlabel('time, ms')
    pl.ylabel('action potential, mV')
    pl.ylim([-20, 120])
    pl.grid('on')
    counter += 1


#np.savetxt('error_derivative_explicit_euler.csv', np.c_[dtArray[1:], error[1:], timeMaxError[1:]], delimiter=',', header='timestep(ms),max_norm_error,time_of_max_norm_error(ms)')
f, (ax1) = pl.subplots(1, 1)
'''ax1.plot(np.log(dtArray[1:]), np.log(error[1:]), 'k-o', linewidth=4, markersize = 15)
ax1.set_xlabel('ln(timestep)')
ax1.set_ylabel('ln(absolute error)')
ax1.grid('on')'''
ax1.plot(dtArray[1:], error[1:], 'k-o', linewidth=4, markersize = 15)
ax1.set_xlabel('timestep, ms')
ax1.set_ylabel('absolute error, V/s')
ax1.grid('on')
# f.figure()

#np.savetxt('error_explicit_euler.csv', np.c_[dtArray[1:], error[1:], timeMaxError[1:]], delimiter=',', header='timestep(ms),max_norm_error,time_of_max_norm_error(ms)')

'''
pl.figure()
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
