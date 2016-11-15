from __future__ import division
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import time
import numba as nb
import math as mt
#from forErrorEtimation import *


font = {'family': 'normal',
        'weight': 'normal',
        'size': 20}

matplotlib.rc('font', **font)

@nb.jit(nopython=True)
def alpha_n(val): return 0.01 * (-val + 10) / (mt.exp((-val + 10) / 10) - 1) if val != 10 else 0.1

@nb.jit(nopython=True)
def beta_n(val):  return 0.125 * mt.exp(-val / 80)

@nb.jit(nopython=True)
def n_inf(val): return alpha_n(val) / (alpha_n(val) + beta_n(val))

@nb.jit(nopython=True)
def alpha_m(val): return (10) * 0.1 * (-val + 25) / (mt.exp((-val + 25) / 10) - 1) if val != 25 else 5

@nb.jit(nopython=True)
def beta_m(val): return (10.)*4 * mt.exp(-val / 18)

@nb.jit(nopython=True)
def m_inf(val): return alpha_m(val) / (alpha_m(val) + beta_m(val))

@nb.jit(nopython=True)
def alpha_h(val): return 5e-1*0.07 * mt.exp(-val / 20)

@nb.jit(nopython=True)
def beta_h(val): return 5e-1*1 / (mt.exp((-val + 30) / 10) + 1)

@nb.jit(nopython=True)
def h_inf(val): return alpha_h(val) / (alpha_h(val) + beta_h(val))


@nb.jit(nopython=True)
def EXP(val):
    return mt.exp(val)


def CalculateNorm(arrayAnalitical, array2):
    k = len(arrayAnalitical) / len(array2)
    normArray = []

    # calculating RRHS error
    numerator, denumerator = 0, 0
    for i in range(len(array2)):
        numerator += (arrayAnalitical[k*i] - array2[i])**2
        denumerator += (arrayAnalitical[k*i])**2

        delta = abs(arrayAnalitical[int(k * i)] - array2[i])
        normArray.append(delta)

    RRHS_error = np.sqrt(numerator / denumerator)

    # calculating max|o| error
    normArray = np.array(normArray)
    max_module = np.amax(normArray)
    amplitude = np.amax(arrayAnalitical) - np.amin(arrayAnalitical)
    indexMax = np.argmax(normArray)

    return list((100 * RRHS_error, 100*max_module / amplitude, 100 * max_module / np.sqrt(( (len(arrayAnalitical))**(-1)) * denumerator)))




g_n = 1200
g_k = 36
g_l = 0.3
v_n = 115
v_k = -12
v_l = 10.613
c = 1

dtArray = np.linspace(1e-5, 1e-3, 10)

print dtArray
T = 10

@nb.jit(nopython=True)
def rhs(i_s, m_, n_, h_, v_, c_):
    return (1. / c_) * (i_s - g_n * (m_) ** 3 * (h_) * (v_ - v_n) - g_k * (n_ ** 4) * (v_ - v_k) - g_l * (v_ - v_l))

@nb.jit(nopython=True)
def rhs_gating_vars(alpha_, beta_, v_, var_):
    return alpha_(v_) * (1 - var_) - beta_(v_) * var_


step_v = 1e-4
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

error, error_e, timeMaxError, timeMaxError_e = [], [], [], []
amplitude = 0
counter = 0
omega = 1 # relaxation parameter

@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], \
                        nb.float64[:]), nopython=True)
def CalculateHHusingExplicitEuler(time_array, size, dt, counter, m, n, h, v, Rhs, I_s):
    m[0] = m_inf(v_rest)
    h[0] = h_inf(v_rest)
    n[0] = n_inf(v_rest)
    v[0] = v_rest

    for i in xrange(1, size):
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1], c)

        dv = Rhs[i-1] * dt
        v[i] = v[i-1] + dv



@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def CalculateHHusingSImplicitEuler(time_array, size, dt, counter, m, n, h, v, Rhs, I_s):

    '''
    v, m, n, h, Rhs = [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)],
    #np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    v_e, m_e, n_e, h_e, Rhs_e = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    I_s = np.zeros(len(time_array))
    '''

    m[0] = m_inf(v_rest)
    h[0] = h_inf(v_rest)
    n[0] = n_inf(v_rest)
    v[0] = v_rest


    dtStar = dt / 2
    for i in xrange(1, size):
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))

        derivative = (rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1] + d_gating_var, c) - rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1], c)) / d_gating_var
        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1], c)

        dv = Rhs[i-1] * dt / (1 - omega * dt * derivative)
        v[i] = v[i-1] + dv



amplitude = 0
start = time.clock()
pl.figure()


timingEE, timingSIE = [], []
for dt in dtArray:
    startMine = time.clock()

    # grid arrays and parameters
    '''
    time_array = np.zeros(SIZE)
    v, m, n, h, Rhs = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    v_e, m_e, n_e, h_e, Rhs_e = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    I_s = np.zeros(len(time_array))
    '''
    time_array = np.arange(0, T, dt)
    SIZE = len(time_array)
    v = 0*np.ones(SIZE)
    m = np.zeros(SIZE)
    n = np.zeros(SIZE)
    h = np.zeros(SIZE)
    Rhs = np.zeros(SIZE) # [0. for i in range(size)]
    v_e = 0 * np.ones(SIZE) #[0. for i in range(size)]
    m_e = np.zeros(SIZE) #[0. for i in range(size)]
    h_e = np.zeros(SIZE) # [0. for i in range(size)]
    n_e = np.zeros(SIZE) #[0. for i in range(size)]
    Rhs_e = np.zeros(SIZE) #[0. for i in range(size)]
    I_s = np.zeros(SIZE) #[10. for i in range(size)]
    I_s[:] = 10


    NUM_LAUNCHES = 10
    startEE = time.clock()
    for i in range(NUM_LAUNCHES):
        CalculateHHusingExplicitEuler(time_array, SIZE, dt, counter, m, n, h, v_e, Rhs, I_s)
    timingEE.append((time.clock() - startEE)/NUM_LAUNCHES)

    startSIE = time.clock()
    for i in range(NUM_LAUNCHES):
        CalculateHHusingSImplicitEuler(time_array, SIZE, dt, counter, m, n, h, v, Rhs, I_s)
    timingSIE.append((time.clock() - startSIE)/NUM_LAUNCHES)


    #time_array = np.array(time_array)
    #V = np.array(time_array)
    #V_e = np.array(time_array)


    if counter == 0:
        analiticalSolution = np.zeros(len(time_array))


    '''
    m[0] = m_e[0] = m_inf(v_rest)
    h[0] = h_e[0] = h_inf(v_rest)
    n[0] = n_e[0] = n_inf(v_rest)
    v[0] = v_e[0] = v_rest

    #I_s = np.zeros(len(time_array))
    I_s[:] = 10

    dtStar = dt / 2
    for i in range(1, len(time_array)):

        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))


        m_e[i] = m_inf(v_e[i-1]) + (m_e[i-1] - m_inf(v_e[i-1])) * np.exp(-dt * (alpha_m(v_e[i-1]) + beta_m(v_e[i-1])))
        n_e[i] = n_inf(v_e[i-1]) + (n_e[i-1] - n_inf(v_e[i-1])) * np.exp(-dt * (alpha_n(v_e[i-1]) + beta_n(v_e[i-1])))
        h_e[i] = h_inf(v_e[i-1]) + (h_e[i-1] - h_inf(v_e[i-1])) * np.exp(-dt * (alpha_h(v_e[i-1]) + beta_h(v_e[i-1])))

        # rhs = (1./c) * (I_s[i-1] - g_n*m**3*h*(v[i-1]-v_n) - g_k*n**4*(v[i-1]-v_k) - g_l*(v[i-1]-v_l))
        derivative = (rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1] + d_gating_var) - rhs(I_s[i - 1], m[i], n[i], h[i],
                                                                              v[i - 1])) / d_gating_var

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1])
        Rhs_e[i-1] = rhs(I_s[i-1], m_e[i-1], n_e[i-1], h_e[i-1], v_e[i-1])

        if counter != 0: # for calculating numerical solutions with various timesteps
            dv = Rhs[i-1] * dt / (1 - omega * dt * derivative)
            dv_e = Rhs_e[i-1] * dt
        else: # for calculating "analitical" solution using Explicit Euler
            dv = Rhs[i-1] * dt
            dv_e = Rhs_e[i-1] * dt

        v[i] = v[i-1] + dv
        v_e[i] = v_e[i-1] + dv_e

        if (i % 2 == 0):
            dtStar = dt / 1
        else:
            dtStar = dt
    '''

    if (counter == 0):
        analiticalSolution[:] = v_e[:]
        amplitude = np.amin(analiticalSolution)
        #print 'ana = ', analiticalSolution
        analiticalSolution[:] -= amplitude


    timeMine = time.clock() - startMine
    #print timeMine

    #print "amplitude %.2e" % amplitude
    v[:] -= amplitude
    v_e[:] -= amplitude


    errTmp = CalculateNorm(analiticalSolution, v)
    errTmp_e = CalculateNorm(analiticalSolution, v_e)

    error.append(errTmp)
    error_e.append(errTmp_e)
    #timeMaxError.append(indexErrTpm * dt)

    if counter == 0:
        pl.plot(time_array, analiticalSolution, '--k', label='analytical (dt = %.2e ms)' % dtArray[counter], linewidth=5)
    elif (counter%3 == 0):
        pl.plot(time_array, v, '-', label='dt = %.2e ms' % dtArray[counter], linewidth=3)
    #pl.title('timestep = %.0e ms\n' % (dt))
    pl.legend(loc='upper right')
    pl.xlabel('time, ms')
    pl.ylabel('action potential, mV')
    #pl.ylim([-20, 120])
    pl.grid('on')
    counter += 1
#pl.show()
print 'time elapsed = %.2e sec' % (time.clock() - start)

def CalculatePolyfit(arrayX, arrayY, order):
    z = np.polyfit(arrayX, arrayY, order)
    p = np.poly1d(z)
    return p # is function of polynom type


left, right = 2, 20
orderOfpolynomial = 1
# calc the trendline
'''for i in range(3):
    ax1.plot(error[i][left:right], CalculatePolyfit(error[i][left:], dtArray[left:], 1)(error[i][left:])[0], 'b-', linewidth=4)
    ax1.plot(error_e[i][left:right], CalculatePolyfit(error_e[i][left:], dtArray[left:], 1)(error_e[i][left:])[0], 'g-', linewidth=4)'''

error = np.array(error)
error_e = np.array(error_e)
#error = map(list, zip(*error))
#error_e = map(list, zip(*error_e))

list_of_errors = [1., 3., 5.]
LEN = len(list_of_errors)
NUM_ERROR_TYPES = 3


dt_for_errors_list = np.array([np.zeros(LEN) for i in range(3)])
dt_for_errors_list_e = np.array([np.zeros(LEN) for i in range(3)])
computational_time_SIE_list = np.array([np.zeros(LEN) for i in range(3)])
computational_time_EE_list = np.array([np.zeros(LEN) for i in range(3)])
speedup_list = np.array([np.zeros(LEN) for i in range(3)])

# loop for error types
for i in range(NUM_ERROR_TYPES):
    dt_for_errors_list[i] = CalculatePolyfit(error[left:right,i], dtArray[left:right], 1)(list_of_errors)
    dt_for_errors_list_e[i] = CalculatePolyfit(error_e[left:right,i], dtArray[left:right], 1)(list_of_errors)

for i in range(NUM_ERROR_TYPES):
    computational_time_SIE_list[i] = CalculatePolyfit(dtArray[left:right], timingSIE[left:right], 1)(dt_for_errors_list[:,i])
    computational_time_EE_list[i] = CalculatePolyfit(dtArray[left:right], timingEE[left:right], 1)(dt_for_errors_list_e[:,i])

np.array(computational_time_SIE_list)
np.array(computational_time_SIE_list)

speedup_list = np.zeros((NUM_ERROR_TYPES, LEN))
for i in range(NUM_ERROR_TYPES):
    for j in range(LEN):
        #speedup_list[i ,j] = computational_time_EE_list[i,j] / computational_time_SIE_list[i, j]
        speedup_list[i ,j] = dt_for_errors_list[i,j] / dt_for_errors_list_e[i,j] / 1.1
#print speedup_list

np.savetxt('speedup_T%.1fsec.csv' % T, np.c_[list_of_errors,\
                                             dt_for_errors_list_e[0,:], dt_for_errors_list[0,:],\
                                             dt_for_errors_list_e[1,:], dt_for_errors_list[1,:],\
                                             dt_for_errors_list_e[2,:], dt_for_errors_list[2,:],\
                                            speedup_list[0,:], speedup_list[1,:], speedup_list[2,:]],\
                                            fmt='%.2e', delimiter=',',\
        header='error,dt_RRMS_EE,dt_RRMS_SIE,dt_maxmod_EE,dt_maxmod_SIE,dt_mixed_EE,dt_mixed_SIE,speedup_RRMS,speedup_maxmod,speedup_mixed')
#error_e[0,1:], error[1,1:], error_e[1,1], error[2,1:], error_e[2,1:]

np.savetxt('errors_3_types_%.1fsec.csv' % T, np.c_[dtArray[1:],\
                                                   error_e[1:,0], error[1:,0],\
                                                   error_e[1:,1], error[1:,1],\
                                                   error_e[1:,2], error[1:,2],\
                                                   timingEE[1:], timingSIE[1:]],\
                                                    fmt='%.2e', delimiter=',',\
           header='timestep(ms),RRMS_EE,RRMS_SIE,max_mod_EE, max_mod_SIE,mixed_EE,mixed_SIE,timingEE,timingSIE')



f, (ax1) = pl.subplots(1, 1)
#ax1.plot(np.log(dtArray[1:]), np.log(error[1:]), 'k-o', linewidth=4, markersize = 15)
#ax1.set_xlabel('ln(timestep)')
#ax1.set_ylabel('ln(absolute error)')
ax1.grid('on')
ax1.plot((dtArray[1:]), (error[1:,1]), 'b-o', label='Simplified Backward Euler', linewidth=4, markersize = 10)
ax1.plot((dtArray[1:]), (error_e[1:,1]), 'g-o', label='Forward Euler', linewidth=4, markersize = 10)
ax1.legend()
'''
ax1.plot((dtArray[1:]), (error[1][1:]), 'b-s', label='Simplified Backward Euler', linewidth=4, markersize = 10)
ax1.plot((dtArray[1:]), (error_e[1][1:]), 'g-s', label='Forward Euler', linewidth=4, markersize = 10)
ax1.grid('on')
#ax2.legend()

ax1.plot((dtArray[1:]), (error[2][1:]), 'b-v', label='Simplified Backward Euler', linewidth=4, markersize = 10)
ax1.plot((dtArray[1:]), (error_e[2][1:]), 'g-v', label='Forward Euler', linewidth=4, markersize = 10)
#ax1.grid('on')
#ax3.legend()
ax1.set_ylim([0, 15])
extraticks = [5]'''
#ax1.set_xticks(list(ax1.xticks()[0]) + extraticks)
ax1.axhline(y=5, linewidth=4, color='r')
#ax2.axhline(y=5, linewidth=4, color='r')
#ax3.axhline(y=5, linewidth=4, color='r')


'''
# calc the trendline
z = np.polyfit(dtArray[1:7], error[1:7], 1)
p = np.poly1d(z)
ax1.plot((dtArray[1:]), p((dtArray[1:])),"b-", linewidth=4)
# the line equation:

ax1.text(0.01, 45e1, "y = (%.2e)*x + (%.2e)" %(z[0], z[1]), color='b', fontsize = 20, fontweight='bold')

# calc the trendline
z_e = np.polyfit(dtArray[1:7], error_e[1:7], 1)
p_e = np.poly1d(z_e)
ax1.plot((dtArray[1:]), p_e((dtArray[1:])),"g-", linewidth=4)
# the line equation:
ax1.text(0.01, 40e1, "y = (%.2e)*x + (%.2e)" %(z_e[0], z_e[1]), color='g', fontsize = 20, fontweight='bold')
'''
ax1.set_xlabel('timestep, ms')
ax1.set_ylabel('RRMS error, %')
ax1.grid('on')
ax1.legend(loc='upper left')
# f.figure()

'''
np.savetxt('errors_3_types_T%.1fsec.csv' % T, np.c_[dtArray[1:], error[0,1:], error_e[0][1:], error[1][1:], error_e[1][1:], error[2][1:], error_e[2][1:], timingSIE[1:], timingEE[1:]], fmt='%.2e', delimiter=',',\
           header='timestep(ms),max_norm_error_SIE,max_norm_error_EE,RRMS_SIE,RRMS_EE,mixed_RRMS_max_norm_SIE,mixed_RRMS_max_norm_EE,timingSIE,timingEE')
'''



pl.show()
