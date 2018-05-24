from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import numba as nb
import math as mt
from scipy import interpolate as intp
import sys


matplotlib.rcParams.update({'font.size': 30})


@nb.jit(nopython=True)
def alpha_n(val): return 0.01 * (-val + 10) / (mt.exp((-val + 10) / 10) - 1) if val != 10 else 0.1

@nb.jit(nopython=True)
def beta_n(val):  return 0.125 * mt.exp(-val / 80)

@nb.jit(nopython=True)
def n_inf(val): return alpha_n(val) / (alpha_n(val) + beta_n(val))

@nb.jit(nopython=True)
def alpha_m(val): return 0.1 * (-val + 25) / (mt.exp((-val + 25) / 10) - 1) if val != 25 else 5

@nb.jit(nopython=True)
def beta_m(val): return 4 * mt.exp(-val / 18)

@nb.jit(nopython=True)
def m_inf(val): return alpha_m(val) / (alpha_m(val) + beta_m(val))

@nb.jit(nopython=True)
def alpha_h(val): return 0.07 * mt.exp(-val / 20)

@nb.jit(nopython=True)
def beta_h(val): return 1 / (mt.exp((-val + 30) / 10) + 1)

@nb.jit(nopython=True)
def h_inf(val): return alpha_h(val) / (alpha_h(val) + beta_h(val))


@nb.jit(nopython=True)
def EXP(val):
    return mt.exp(val)


def CalculateNorm(analiticalSolutionFunc, numericalSolutionArr, dt, amplitude):
    normArray = []

    # calculating RRHS error
    numerator, denumerator = 0, 0
    for i in range(len(numericalSolutionArr)):
        t = i*dt
        numerator += (analiticalSolutionFunc(t) - numericalSolutionArr[i])**2
        denumerator += (analiticalSolutionFunc(t))**2

        delta = abs(analiticalSolutionFunc(t) - numericalSolutionArr[i])
        normArray.append(delta)

    RRHSerror = np.sqrt(numerator / denumerator)

    # calculating max|o| error
    normArray = np.array(normArray)
    maxModule = np.amax(normArray)
    indexMax = np.argmax(normArray)
    errorsList = [100 * RRHSerror, 100*maxModule / amplitude]

    for i in range(len(errorsList)):
        if np.isnan(errorsList[i]) == True:
            errorsList[i] = 1e20 #sys.float_info.max

    return errorsList



g_n = 1200#120
g_k = 36
g_l = 0.3
v_n = 115
v_k = -12
v_l = 10.613
c = 1

T = 8
numPointsArray = np.array([int(2**n) for n in range(5, 17)])
dtArray = np.array([1e-5] + list(float(T)/numPointsArray[::-1])) #[1e-5] + [0.009232069419105898, 0.0041864984602040045] #np.array([1e-5] + list(float(T)/numPointsArray[::-1]))
print dtArray

@nb.jit(nopython=True)
def rhs(i_s, m_, n_, h_, v_, c_):
    return (1. / c_) * (i_s - g_n * (m_) ** 3 * (h_) * (v_ - v_n) - g_k * (n_ ** 4) * (v_ - v_k) - g_l * (v_ - v_l))

@nb.jit(nopython=True)
def rhs_gating_vars(alpha_, beta_, v_, var_):
    return alpha_(v_) * (1 - var_) - beta_(v_) * var_


step_v = 1e-4
d_gating_var = 1e-7
v_rest = 0

error, error_e, timeMaxError, timeMaxError_e = [], [], [], []
amplitude = 0
counter = 0
omega = 1 # relaxation parameter

#@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
 #                       nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], \
  #                      nb.float64[:]), nopython=True)
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



#@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
#                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
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

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1], c)
        derivative = (rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1] + d_gating_var, c) - Rhs[i-1]) / d_gating_var

        dv = Rhs[i-1] * dt / (1 - omega * dt * derivative)
        v[i] = v[i-1] + dv


analiticalSolutionFunc = None
amplitude = 0
analiticalSolution = None
start = time.clock()

plt.figure()
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
    print 'Iteration #%d' % counter
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


    NUM_LAUNCHES = 1
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
        MIN, MAX = np.amin(analiticalSolution), np.amax(analiticalSolution)
        amplitude = MAX - MIN
        analiticalSolution[:] -= MIN
        analiticalSolutionFunc = intp.interp1d(time_array, analiticalSolution)


    timeMine = time.clock() - startMine

    v[:] -= MIN
    v_e[:] -= MIN

    errTmp = CalculateNorm(analiticalSolutionFunc, v, dt, amplitude)
    errTmp_e = CalculateNorm(analiticalSolutionFunc, v_e, dt, amplitude)

    error.append(errTmp)
    error_e.append(errTmp_e)
    #timeMaxError.append(indexErrTpm * dt)

    if counter == 0:
        plt.plot(time_array, analiticalSolution, 'k-', label='gNa=1200 mS/cm^2', linewidth=4)
    #if counter == 1:
    #    plt.plot(time_array, v, '-', label='5% RRMS error', linewidth=2)
    #if counter == 2:
    #    plt.plot(time_array, v, '-', label='5% Maxmod error', linewidth=2)

    #elif (counter%4 == 0):
        #plt.plot(time_array, v, '-', label='dt = %.2e ms' % dtArray[counter], linewidth=3)
    #pl.title('timestep = %.0e ms\n' % (dt))
    plt.legend(loc='best')
    plt.xlabel('Time, ms')
    plt.ylabel('Membrane potential, mV')
    plt.ylim([0, 140])
    plt.grid('on')
    counter += 1
plt.show()
print 'time elapsed = %.2e sec' % (time.clock() - start)

error = np.array(error)
error_e = np.array(error_e)


def CalculatePolyfit(arrayX, arrayY):
    p = intp.interp1d(arrayX, arrayY)
    return p #

dtSpecificList = []
for i in range(2):
    dtSpecificList.append(CalculatePolyfit(error[1:, i], dtArray[1:])(5))
print 'Timesteps for 5%', dtSpecificList




'''
left, right = 2, 9
orderOfpolynomial = 1
# calc the trendline
for i in range(3):
    ax1.plot(error[i][left:right], CalculatePolyfit(error[i][left:], dtArray[left:], 1)(error[i][left:])[0], 'b-', linewidth=4)
    ax1.plot(error_e[i][left:right], CalculatePolyfit(error_e[i][left:], dtArray[left:], 1)(error_e[i][left:])[0], 'g-', linewidth=4)

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
    computational_time_SIE_list[i] = CalculatePolyfit(dtArray[left:right], timingSIE[left:right], 1)(dt_for_errors_list[i,:])
    computational_time_EE_list[i] = CalculatePolyfit(dtArray[left:right], timingEE[left:right], 1)(dt_for_errors_list_e[i,:])

computational_time_EE_list = np.array(computational_time_EE_list)
computational_time_SIE_list = np.array(computational_time_SIE_list)

speedup_list = np.zeros((NUM_ERROR_TYPES, LEN))
for i in range(NUM_ERROR_TYPES):
    for j in range(LEN):
        #speedup_list[i ,j] = computational_time_EE_list[i,j] / computational_time_SIE_list[i, j]
        speedup_list[i ,j] = dt_for_errors_list[i,j] / dt_for_errors_list_e[i,j] / 1.1
#print speedup_list



np.savetxt('data_T%.1fsec.csv' % T, np.c_[list_of_errors,\
                                             dt_for_errors_list_e[0,:], dt_for_errors_list[0,:],\
                                             dt_for_errors_list_e[1,:], dt_for_errors_list[1,:],\
                                             dt_for_errors_list_e[2,:], dt_for_errors_list[2,:],\
                                            computational_time_EE_list[0,:], computational_time_SIE_list[0,:],\
                                            computational_time_EE_list[1,:], computational_time_SIE_list[1,:],\
                                            computational_time_EE_list[2,:], computational_time_SIE_list[2,:]],\
                                            fmt='%.2e', delimiter=',',\
        header='error,dt_RRMS_EE,dt_RRMS_SIE,dt_maxmod_EE,dt_maxmod_SIE,dt_mixed_EE,dt_mixed_SIE,CPU_time_RRMS_EE,CPU_time_RRMS_SIE,CPU_time_maxmod_EE,CPU_time_maxmod_SIE,CPU_time_mixed_EE,CPU_time_mixed_SIE')
#error_e[0,1:], error[1,1:], error_e[1,1], error[2,1:], error_e[2,1:]

np.savetxt('errors_3_types_%.1fsec.csv' % T, np.c_[dtArray[1:],\
                                                   error_e[1:,0], error[1:,0],\
                                                   error_e[1:,1], error[1:,1],\
                                                   error_e[1:,2], error[1:,2],\
                                                   timingEE[1:], timingSIE[1:]],\
                                                    fmt='%.2e', delimiter=',',\
           header='timestep(ms),RRMS_EE,RRMS_SIE,max_mod_EE, max_mod_SIE,mixed_EE,mixed_SIE,timingEE,timingSIE')

'''

fig2 = plt.figure()
#ax1.plot(np.log(dtArray[1:]), np.log(error[1:]), 'k-o', linewidth=4, markersize = 15)
#ax1.set_xlabel('ln(timestep)')
#ax1.set_ylabel('ln(absolute error)')
plt.loglog((timingSIE[1:]), (error[1:,0]), 'b-o', label='Simplified B.Euler', linewidth=4, markersize = 10)
plt.loglog((timingEE[1:]), (error_e[1:,0]), 'g-o', label='F.Euler', linewidth=4, markersize = 10)
plt.grid('on')
plt.xlabel('t calc, s')
plt.ylabel('RRMS error, %')
plt.ylim([0, 100])
plt.axhline(y=5, linewidth=4, color='k', linestyle='--', label='5%')
plt.axhline(y=1, linewidth=4, color='k', linestyle='--', label='1%')
plt.legend(loc='best')


fig3 = plt.figure()
plt.loglog((timingSIE[1:]), (error[1:,1]), 'b-o', label='Simplified B.Euler', linewidth=4, markersize = 10)
plt.loglog((timingEE[1:]), (error_e[1:,1]), 'g-o', label='F.Euler', linewidth=4, markersize = 10)
plt.grid('on')
plt.xlabel('t calc, s')
plt.ylabel('Maxmod error, %')
plt.ylim([0., 100])
plt.axhline(y=5, linewidth=4, color='k', linestyle='--', label='5%')
plt.axhline(y=1, linewidth=4, color='k', linestyle='--', label='1%')
plt.legend(loc='best')

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
extraticks = [5]
#ax1.set_xticks(list(ax1.xticks()[0]) + extraticks)
'''

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

ax1.set_xlabel('Timestep, ms')
ax1.set_ylabel('Mixed norm error, %')
ax1.grid('on')
ax1.legend(loc='upper left')
# f.figure()


np.savetxt('errors_3_types_T%.1fsec.csv' % T, np.c_[dtArray[1:], error[0,1:], error_e[0][1:], error[1][1:], error_e[1][1:], error[2][1:], error_e[2][1:], timingSIE[1:], timingEE[1:]], fmt='%.2e', delimiter=',',\
           header='timestep(ms),max_norm_error_SIE,max_norm_error_EE,RRMS_SIE,RRMS_EE,mixed_RRMS_max_norm_SIE,mixed_RRMS_max_norm_EE,timingSIE,timingEE')
'''
plt.show()
