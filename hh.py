from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import numba as nb
import math as mt
from scipy import interpolate as intp
import sys


matplotlib.rcParams.update({'font.size': 25})


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
            errorsList[i] = sys.float_info.max

    return errorsList



g_n = [120.] + [float(2*i) for i in range(100, 401, 100)]
g_k = 36
g_l = 0.3
v_n = 115
v_k = -12
v_l = 10.613
c = 1

T = 8
numPointsArray = np.array([int(2**n) for n in range(5, 17)])
dtArray = np.array(([1e-4]) + list(float(T)/numPointsArray[::-1])) #[1e-5] + [0.009232069419105898, 0.0041864984602040045] #np.array([1e-5] + list(float(T)/numPointsArray[::-1]))
print (dtArray)

@nb.jit(nopython=True)
def rhs(i_s, m_, n_, h_, v_, c_, g_n):
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

@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], \
                        nb.float64[:], nb.float64), nopython=True)
def CalculateHHusingExplicitEuler(time_array, size, dt, counter, m, n, h, v, Rhs, I_s, g_n):
    m[0] = m_inf(v_rest)
    h[0] = h_inf(v_rest)
    n[0] = n_inf(v_rest)
    v[0] = v_rest

    for i in range(1, size):
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1], c, g_n)

        dv = Rhs[i-1] * dt
        v[i] = v[i-1] + dv



@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64), nopython=True)
def CalculateHHusingSImplicitEuler(time_array, size, dt, counter, m, n, h, v, Rhs, I_s, g_n):

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
    for i in range(1, size):
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1], c, g_n)
        derivative = (rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1] + d_gating_var, c, g_n) - Rhs[i-1]) / d_gating_var

        dv = Rhs[i-1] * dt / (1 - omega * dt * derivative)
        v[i] = v[i-1] + dv
#hh_analytical_solutions

analiticalSolutionFunc = None
amplitude = 0
analiticalSolution = None
start = time.clock()
speedupRRMS, speedupMaxmod = [], []
ERROR = 1.
counterOuter = 0

plt.figure()
for stiffnessParameter in g_n:
    timingEE, timingSIE = [], []
    error, error_e = [], []
    counterInner = 0
    for dt in dtArray:
        startMine = time.clock()

        # grid arrays and parameters
        '''
        time_array = np.zeros(SIZE)
        v, m, n, h, Rhs = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
        v_e, m_e, n_e, h_e, Rhs_e = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
        I_s = np.zeros(len(time_array))
        '''
        print ('Iteration #%d' % counterInner)
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


        NUM_LAUNCHES = 5
        startEE = time.clock()
        for i in range(NUM_LAUNCHES):
            CalculateHHusingExplicitEuler(time_array, SIZE, dt, counter, m, n, h, v_e, Rhs, I_s, stiffnessParameter)
        timingEE.append((time.clock() - startEE)/NUM_LAUNCHES)

        startSIE = time.clock()
        for i in range(NUM_LAUNCHES):
            CalculateHHusingSImplicitEuler(time_array, SIZE, dt, counter, m, n, h, v, Rhs, I_s, stiffnessParameter)
        timingSIE.append((time.clock() - startSIE)/NUM_LAUNCHES)

        if counter == 0:
            analiticalSolution = np.zeros(len(time_array))

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

        if counterInner == 0:
            if counterOuter == 0:
                plt.plot(time_array, analiticalSolution, 'k--', label=r'$g_{Na}$ = %d $мСм/см^2$' % stiffnessParameter, linewidth=4)
            else:
                plt.plot(time_array, analiticalSolution, '-', label=r'$g_{Na}$ = %d $мСм/см^2$' % stiffnessParameter, linewidth=4)
        #if counter == 1:
        #    plt.plot(time_array, v, '-', label='5% RRMS error', linewidth=2)
        #if counter == 2:
        #    plt.plot(time_array, v, '-', label='5% Maxmod error', linewidth=2)

        #elif (counter%4 == 0):
            #plt.plot(time_array, v, '-', label='dt = %.2e ms' % dtArray[counter], linewidth=3)
        #pl.title('timestep = %.0e ms\n' % (dt))
        plt.legend(loc='best', prop={'size': 20})
        plt.xlabel('Время, мс')
        plt.ylabel('V, мВ')
        plt.ylim([0, 140])
        plt.xlim([0, 8])
        plt.grid('on')
        counterInner += 1

    counterOuter += 1
    error_for_calc = np.array(error)
    error_e_for_calc = np.array(error_e)

    speedupRRMS.append([intp.interp1d(error_e_for_calc[1:, 0], timingEE[1:])(5.) / \
                       intp.interp1d(error_for_calc[1:, 0], timingSIE[1:])(5.), \
                        intp.interp1d(error_e_for_calc[1:, 0], timingEE[1:])(1.) / \
                        intp.interp1d(error_for_calc[1:, 0], timingSIE[1:])(1.)]
                       )


    speedupMaxmod.append([intp.interp1d(error_e_for_calc[1:, 1], timingEE[1:])(5.) / \
                       intp.interp1d(error_for_calc[1:, 1], timingSIE[1:])(5.), \
                          intp.interp1d(error_e_for_calc[1:, 1], timingEE[1:])(1.) / \
                          intp.interp1d(error_for_calc[1:, 1], timingSIE[1:])(1.)]
                         )

plt.show()
print ('time elapsed = %.2e sec' % (time.clock() - start))

speedupRRMS = np.array(speedupRRMS)
speedupMaxmod = np.array(speedupMaxmod)

fig2 = plt.figure()
plt.plot(g_n, speedupRRMS[:, 0], 'b-o', label='5% RRMS', linewidth=4, markersize=10)
plt.plot(g_n, speedupRRMS[:, 1], 'g-o', label='1% RRMS', linewidth=4, markersize=10)
plt.plot(g_n, speedupMaxmod[:, 0], 'r-o', label='5% Maxmod', linewidth=4, markersize=10)
plt.plot(g_n, speedupMaxmod[:, 1], 'y-o', label='1% Maxmod', linewidth=4, markersize=10)
plt.xlabel(r'$g_{Na}, мСм/см^2$')
plt.ylabel('Ускорение')
plt.grid('on')
plt.legend(loc='best', prop={'size': 15})
fig2.tight_layout()
plt.show()

#error = np.array(error)
#error_e = np.array(error_e)

def CalculatePolyfit(arrayX, arrayY):
    p = intp.interp1d(arrayX, arrayY)
    return p #

#dtSpecificList = []
#for i in range(2):
#    dtSpecificList.append(CalculatePolyfit(error[1:, i], dtArray[1:])(5))
#print 'Timesteps for 5%', dtSpecificList



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

