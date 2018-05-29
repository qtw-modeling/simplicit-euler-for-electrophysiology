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


k = 8. #[float(i) for i in range(5, 15)]
a = 0.15

u_rest = 0
v_rest = 0


T = 40.
numPointsArray = np.array([int(2**n) for n in range(5, 15)])
dtArray = np.array([1e-4] + [0.06792465, 0.02008577]) #list(float(T)/numPointsArray[::-1])) #[2**(-n) for n in range(17, 0, -1)] #16 points - for error estimation

print(dtArray)

@nb.jit(nopython=True)
def eps(u, v):
    eps0 = 0.002
    mu1 = 0.2
    mu2 = 0.3
    return eps0 + (mu1*v) / (u + mu2)

@nb.jit(nopython=True)
def rhsU(u, v, k, iStim=0.1):
    return -k*u*(u - a)*(u - 1) - u*v + iStim

@nb.jit(nopython=True)
def rhsV(u, v, k):
    return eps(u, v)*(-v - k*u*(u - a - 1))

#error_maxmod_vs_tcalc_mod_hh

stepForDer = 1e-2
d_gating_var = 1e-7


error, error_e, timeMaxError, timeMaxError_e = [], [], [], []
amplitude = 0
counter = 0
omega = 1 # relaxation parameter

@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], \
                        nb.float64[:], nb.float64), nopython=True)
def CalculateUsingExplicitEuler(time_array, size, dt, counter, u, v, RhsU, RhsV, I_s, k):
    u[0] = u_rest
    v[0] = v_rest

    for i in range(1, size):
        RhsU[i - 1] = rhsU(u[i - 1], v[i - 1], k)
        RhsV[i - 1] = rhsV(u[i - 1], v[i - 1], k)

        u[i] = u[i - 1] + dt*RhsU[i - 1]
        v[i] = v[i - 1] + dt*RhsV[i - 1]



@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int64, \
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],\
                        nb.float64[:], nb.float64), nopython=True)
def CalculateUsingSImplicitEuler(time_array, size, dt, counter, u, v, RhsU, RhsV, I_s, k):

    '''
    v, m, n, h, Rhs = [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)], [0. for i in range(size)],
    #np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    v_e, m_e, n_e, h_e, Rhs_e = np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array)), np.zeros(len(time_array))
    I_s = np.zeros(len(time_array))
    '''

    u[0] = u_rest
    v[0] = v_rest

    for i in range(1, size):
        RhsU[i - 1] = rhsU(u[i - 1], v[i - 1], k)
        RhsV[i - 1] = rhsV(u[i - 1], v[i - 1], k)

        dRhsUdU = (rhsU(u[i - 1] + stepForDer, v[i - 1], k) \
                                - RhsU[i - 1]) / stepForDer

        v[i] = v[i - 1] + dt*RhsV[i - 1]
        u[i] = u[i - 1] + dt*RhsU[i - 1] / (1 - dt*dRhsUdU)





amplitude = 0
start = time.clock()
plt.figure()

stiffnessCoeff = k
ERROR = 5 # in %
speedup = []
#for stiffnessCoeff in k:
timingEE, timingSIE = [], []
error, error_e = [], []
counter = 0
for dt in dtArray:
    #startMine = time.clock()

    print ('Iteration #%d' % counter)
    time_array = np.arange(0., T, dt)
    SIZE = len(time_array)
    u = np.zeros(SIZE)
    v = np.ones(SIZE)
    RhsU = np.zeros(SIZE) # [0. for i in range(size)]
    RhsV = np.zeros(SIZE)

    u_e = np.zeros(SIZE)
    v_e = np.zeros(SIZE) #[0. for i in range(size)]

    RhsU_e = np.zeros(SIZE) #[0. for i in range(size)]
    RhsV_e = np.zeros(SIZE)

    I_s =10*np.ones(SIZE) #[10. for i in range(size)]



    NUM_LAUNCHES = 5
    startEE = time.clock()
    for i in range(NUM_LAUNCHES):
        CalculateUsingExplicitEuler(time_array, SIZE, dt, counter, u_e, v_e, RhsU_e, RhsV_e, I_s, \
                                                                                        k)
    timingEE.append((time.clock() - startEE)/NUM_LAUNCHES)

    startSIE = time.clock()
    for i in range(NUM_LAUNCHES):
        CalculateUsingSImplicitEuler(time_array, SIZE, dt, counter, u, v, RhsU, RhsV, I_s, \
                                                                                        k)
    timingSIE.append((time.clock() - startSIE)/NUM_LAUNCHES)

    if counter == 0:
        analiticalSolution = np.zeros(len(time_array))


    if (counter == 0):
        analiticalSolution[:] = 100*u_e[:] - 80 # scaled to membrane voltage
        MIN, MAX = np.amin(analiticalSolution), np.amax(analiticalSolution)
        amplitude = MAX - MIN
        analiticalSolution[:] -= MIN
        analiticalSolutionFunc = intp.interp1d(time_array, analiticalSolution)


    u_e[:] = 100*u_e[:] - 80 # scaled to membrane voltage
    u[:] = 100*u[:] - 80 # scaled to membrane voltage

    u[:] -= MIN
    u_e[:] -= MIN
    time_array *= 12.9


    errTmp = CalculateNorm(analiticalSolutionFunc, u, dt, amplitude)
    errTmp_e = CalculateNorm(analiticalSolutionFunc, u_e, dt, amplitude)

    error.append(errTmp)
    error_e.append(errTmp_e)

    if counter == 0:
        plt.plot(time_array, analiticalSolution, 'k--', label='Аналитическое', linewidth=2)
    if counter == 1:
        plt.plot(time_array, u, '-', label='Погрешность 5% по норме RRMS', linewidth=2)
    if counter == 2:
        plt.plot(time_array, u, '-', label='Погрешность 5% по норме Maxmod', linewidth=2)
    plt.legend(loc='best', prop={'size': 20})
    plt.xlabel('Время, мс')
    plt.ylabel('V, мВ')
    plt.ylim([0, 140])
    plt.xlim([0, 500])
    plt.grid('on')
    counter += 1

    #error_for_calc = np.array(error)
    #error_e_for_calc = np.array(error_e)

    #speedup.append(intp.interp1d(error_e_for_calc[1:, 0], timingEE[1:])(ERROR) / \
    #               intp.interp1d(error_for_calc[1:, 0], timingSIE[1:])(ERROR))

error = np.array(error)
error_e = np.array(error_e)

'''dtSpecific = [intp.interp1d(error[1:, 0], dtArray[1:])(5.), intp.interp1d(error[1:, 1], dtArray[1:])(5.)]

print('dtSpecific = ', dtSpecific)'''

print ('time elapsed = %.2e sec' % (time.clock() - start))

def CalculatePolyfit(arrayX, arrayY, order):
    p = intp.interp1d(arrayX, arrayY)

    #z = np.polyfit(arrayX, arrayY, order)
    #p = np.poly1d(z)
    return p # is function of polynom type



#ap_analytical_vs_5percent_errors



left, right = 11, 14
orderOfpolynomial = 1
# calc the trendline
'''for i in range(3):
    ax1.plot(error[i][left:right], CalculatePolyfit(error[i][left:], dtArray[left:], 1)(error[i][left:])[0], 'b-', linewidth=4)
    ax1.plot(error_e[i][left:right], CalculatePolyfit(error_e[i][left:], dtArray[left:], 1)(error_e[i][left:])[0], 'g-', linewidth=4)'''

error = np.array(error)
error_e = np.array(error_e)

#print error
#error = map(list, zip(*error))
#error_e = map(list, zip(*error_e))

list_of_errors = [1., 3., 5.]
LEN = len(list_of_errors)
NUM_ERROR_TYPES = 3

'''
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
plt.loglog((timingSIE[1:]), (error[1:, 0]), 'b-o', label='Simplified B.Euler', linewidth=4, markersize=10)
plt.loglog((timingEE[1:]), (error_e[1:, 0]), 'g-o', label='F.Euler', linewidth=4, markersize=10)
plt.grid('on')
plt.xlabel('t calc, s')
plt.ylabel('RRMS error, %')
plt.ylim([0., 100.])
plt.axhline(y=5, linewidth=4, color='k', linestyle='--', label='5%')
plt.axhline(y=1, linewidth=4, color='k', linestyle='--', label='1%')
plt.legend()

fig3 = plt.figure()
plt.loglog((timingSIE[1:]), (error[1:, 1]), 'b-o',label='Simplified B.Euler', linewidth=4, markersize=10)
plt.loglog((timingEE[1:]), (error_e[1:, 1]), 'g-o',label='F.Euler', linewidth=4, markersize=10)
plt.grid('on')
plt.xlabel('t calc, s')
plt.ylabel('Maxmod error, %')
plt.ylim([0., 100.])
plt.axhline(y=5, linewidth=4, color='k', linestyle='--', label='5%')
plt.axhline(y=1, linewidth=4, color='k', linestyle='--', label='1%')
plt.legend()





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


#ax2.axhline(y=5, linewidth=4, color='r')
#ax3.axhline(y=5, linewidth=4, color='r')


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
