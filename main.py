from __future__ import division
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import time
import numba as nb
import math as mt


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
def alpha_m(val): return (1) * 0.1 * (-val + 25) / (mt.exp((-val + 25) / 10) - 1) if val != 25 else 5

@nb.jit(nopython=True)
def beta_m(val): return (1.)*4 * mt.exp(-val / 18)

@nb.jit(nopython=True)
def m_inf(val): return alpha_m(val) / (alpha_m(val) + beta_m(val))

@nb.jit(nopython=True)
def alpha_h(val): return 0.07 * mt.exp(-val / 20)

@nb.jit(nopython=True)
def beta_h(val): return 1 / (mt.exp((-val + 30) / 10) + 1)

@nb.jit(nopython=True)
def h_inf(val): return alpha_h(val) / (alpha_h(val) + beta_h(val))





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

    return 100 * RRHS_error, 100*max_module / amplitude, 100 * max_module / np.sqrt(( (len(arrayAnalitical))**(-1)) * denumerator)




g_n = 1200
g_k = 36
g_l = 0.3
v_n = 115
v_k = -12
v_l = 10.613
c = 1

dtArray = np.linspace(5e-6, 1e-2, 50)

print dtArray
T = 5.0

@nb.jit(nopython=True)
def rhs(i_s, m_, n_, h_, v_):
    return (1. / c) * (i_s - g_n * (m_) ** 3 * (h_) * (v_ - v_n) - g_k * (n_ ** 4) * (v_ - v_k) - g_l * (v_ - v_l))

@nb.jit(nopython=True)
def rhs_gating_vars(alpha, beta, V, var):
    return alpha * (1 - var) - beta * var


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


@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int32, \
                       nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], \
                      nb.float64[:]), nopython=True)
def CalculateHHusingExplicitEuler(time_array, size, dt, counter, m, n, h, v, Rhs, I_s):
    m[0] = m_inf(v_rest)
    h[0] = h_inf(v_rest)
    n[0] = n_inf(v_rest)
    v[0] = v_rest

    dtStar = dt / 2.
    for i in range(1, size):
        '''
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))
        '''

        vPrevious = v[i-1]
        derivative_m = 1.0 / d_gating_var * (rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1] + d_gating_var) -  rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1]))
        derivative_n = 1.0 / d_gating_var * (rhs_gating_vars(alpha_n(vPrevious), beta_n(vPrevious), vPrevious, n[i-1] + d_gating_var) -  rhs_gating_vars(alpha_n(vPrevious), beta_n(vPrevious), vPrevious, n[i-1]))
        derivative_h = 1.0 / d_gating_var * (rhs_gating_vars(alpha_h(vPrevious), beta_h(vPrevious), vPrevious, h[i-1] + d_gating_var) -  rhs_gating_vars(alpha_h(vPrevious), beta_h(vPrevious), vPrevious, h[i-1]))

        m[i] = m[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1]) * dt / (1 - dtStar * derivative_m)
        n[i] = n[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, n[i-1]) * dt / (1 - dtStar * derivative_n)
        h[i] = h[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, h[i-1]) * dt / (1 - dtStar * derivative_h)

        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1])

        dv = Rhs[i-1] * dt
        v[i] = v[i-1] + dv


@nb.jit((nb.float64[:], nb.int64, nb.float64, nb.int32, \
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
    for i in range(1, size):
        vPrevious = v[i-1]
        derivative_m = 1.0 / d_gating_var * (rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1] + d_gating_var) -  rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1]))
        derivative_n = 1.0 / d_gating_var * (rhs_gating_vars(alpha_n(vPrevious), beta_n(vPrevious), vPrevious, n[i-1] + d_gating_var) -  rhs_gating_vars(alpha_n(vPrevious), beta_n(vPrevious), vPrevious, n[i-1]))
        derivative_h = 1.0 / d_gating_var * (rhs_gating_vars(alpha_h(vPrevious), beta_h(vPrevious), vPrevious, h[i-1] + d_gating_var) -  rhs_gating_vars(alpha_h(vPrevious), beta_h(vPrevious), vPrevious, h[i-1]))

        m[i] = m[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, m[i-1]) * dt / (1 - dtStar * derivative_m)
        n[i] = n[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, n[i-1]) * dt / (1 - dtStar * derivative_n)
        h[i] = h[i-1] + rhs_gating_vars(alpha_m(vPrevious), beta_m(vPrevious), vPrevious, h[i-1]) * dt / (1 - dtStar * derivative_h)
        '''
        m[i] = m_inf(v[i-1]) + (m[i-1] - m_inf(v[i-1])) * np.exp(-dt * (alpha_m(v[i-1]) + beta_m(v[i-1])))
        n[i] = n_inf(v[i-1]) + (n[i-1] - n_inf(v[i-1])) * np.exp(-dt * (alpha_n(v[i-1]) + beta_n(v[i-1])))
        h[i] = h_inf(v[i-1]) + (h[i-1] - h_inf(v[i-1])) * np.exp(-dt * (alpha_h(v[i-1]) + beta_h(v[i-1])))
        '''

        derivative = (rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1] + d_gating_var) - rhs(I_s[i - 1], m[i], n[i], h[i], v[i - 1])) / d_gating_var
        Rhs[i-1] = rhs(I_s[i-1], m[i-1], n[i-1], h[i-1], v[i-1])

        dv = Rhs[i-1] * dt / (1 - omega * dt * derivative)
        v[i] = v[i-1] + dv




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
    v = np.zeros(SIZE)
    v_e = np.zeros(SIZE)
    m = np.zeros(SIZE)
    n = np.zeros(SIZE)
    h = np.zeros(SIZE)
    Rhs = np.zeros(SIZE) # [0. for i in range(size)]
    v_e = np.zeros(SIZE) #[0. for i in range(size)]
    m_e = np.zeros(SIZE) #[0. for i in range(size)]
    h_e = np.zeros(SIZE) # [0. for i in range(size)]
    n_e = np.zeros(SIZE) #[0. for i in range(size)]
    Rhs_e = np.zeros(SIZE) #[0. for i in range(size)]
    I_s = np.zeros(SIZE) #[10. for i in range(size)]


    startEE = time.clock()
    CalculateHHusingExplicitEuler(time_array, SIZE, dt, counter, m, n, h, v_e, Rhs, I_s)
    timingEE.append(time.clock() - startEE)

    startSIE = time.clock()
    CalculateHHusingSImplicitEuler(time_array, SIZE, dt, counter, m, n, h, v, Rhs, I_s)
    timingSIE.append(time.clock() - startSIE)


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
        analiticalSolution[:] = v[:]
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

print 'time elapsed = %.2e sec' % (time.clock() - start)
error = map(list, zip(*error))
error_e = map(list, zip(*error_e))

speedup = np.array(timingEE[:]) * (np.array(timingSIE[:]))**(-1)
np.savetxt('errors_3_types.csv', np.c_[dtArray[1:], error[0][1:], error_e[0][1:], error[1][1:], error_e[1][1:], error[2][1:], error_e[2][1:], timingSIE[1:], timingEE[1:]], fmt='%.2e', delimiter=',',\
           header='timestep(ms),max_norm_error_SIE,max_norm_error_EE,RRMS_SIE,RRMS_EE,mixed_RRMS_max_norm_SIE,mixed_RRMS_max_norm_EE,timingSIE,timingEE')



f, (ax1) = pl.subplots(1, 1)
#ax1.plot(np.log(dtArray[1:]), np.log(error[1:]), 'k-o', linewidth=4, markersize = 15)
#ax1.set_xlabel('ln(timestep)')
#ax1.set_ylabel('ln(absolute error)')
ax1.grid('on')
ax1.plot((dtArray[1:]), (error[0][1:]), 'b-o', label='Simplified Backward Euler', linewidth=4, markersize = 10)
ax1.plot((dtArray[1:]), (error_e[0][1:]), 'g-o', label='Forward Euler', linewidth=4, markersize = 10)
ax1.legend()

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
ax1.axhline(y=5, linewidth=4, color='r')
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
'''
ax1.set_xlabel('timestep, ms')
ax1.set_ylabel('RRMS error, %')
ax1.grid('on')
ax1.legend(loc='upper left')
# f.figure()

#np.savetxt('error_explicit_euler.csv', np.c_[dtArray[1:], error[1:], timeMaxError[1:]], delimiter=',', header='timestep(ms),max_norm_error,time_of_max_norm_error(ms)')
'''
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
#pl.show()
