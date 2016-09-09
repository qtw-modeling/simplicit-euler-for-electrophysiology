import scipy as sp
import matplotlib.pylab as plt
from scipy.integrate import odeint
from scipy import stats
import scipy.linalg as lin
from forErrorEtimation import *

## Full Hodgkin-Huxley Model (copied from Computational Lab 2)

# Constants
C_m  =   1.0 # membrane capacitance, in uF/cm^2
g_Na = 1200.0 # maximum conducances, in mS/cm^2
g_K  =  36.0
g_L  =   0.3
E_Na =  115.0 # Nernst reversal potentials, in mV
E_K  = -12.0
E_L  = 10.613
E_rest = 0.0

# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(val): return 0.1 * (-val + 25) / (sp.exp((-val + 25) / 10) - 1) if val != 25 else 5
def beta_m(val):  return 4 * sp.exp(-val / 18)
def alpha_h(val): return 0.07 * sp.exp(-val / 20)
def beta_h(val):  return 1 / (sp.exp((-val + 30) / 10) + 1)
def alpha_n(val): return 0.01 * (-val + 10) / (sp.exp((-val + 10) / 10) - 1) if val != 10 else 0.1
def beta_n(val):  return 0.125 * sp.exp(-val / 80)

# Membrane currents (in uA/cm^2)
#  Sodium (Na = element name)
def I_Na(V,m,h):return g_Na * m**3 * h * (V - E_Na)
#  Potassium (K = element name)
def I_K(V, n):  return g_K  * n**4     * (V - E_K)
#  Leak
def I_L(V):     return g_L             * (V - E_L)

# External current
def I_inj(t): # step up 10 uA/cm^2 every 100ms for 400ms
    return 10#*(t>100) - 10*(t>200) + 35*(t>300)
    #return 10*t

# Integrate!
def dALLdt(X, t):
    V, m, h, n = X

    #calculate membrane potential & activation variables
    dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dmdt, dhdt, dndt



def n_inf(val): return alpha_n(val) / (alpha_n(val) + beta_n(val))
def m_inf(val): return alpha_m(val) / (alpha_m(val) + beta_m(val))
def h_inf(val): return alpha_h(val) / (alpha_h(val) + beta_h(val))



# Time to integrate till
T = 5.0 # in msec
dtArray = sp.linspace(1e-4, 1e-2, 20)
error = []




plt.figure()
counter = 0
for timestep in dtArray:
    t = sp.arange(0.0, T, timestep) # time array to integrate over

    X = odeint(dALLdt, [E_rest, m_inf(E_rest), h_inf(E_rest), n_inf(E_rest)], t)
    V = X[:,0]
    m = X[:,1]
    h = X[:,2]
    n = X[:,3]
    ina = I_Na(V,m,h)
    ik = I_K(V, n)
    il = I_L(V)

    if counter == 0:
        analyticalSolution = sp.zeros(len(V))
        analyticalSolution[:] = V[:]
    counter += 1

    error.append(CalculateNorm(analyticalSolution, V))


    #plt.subplot(1, 1, 1)
    plt.title('Hodgkin-Huxley Neuron')
    plt.plot(t, V, linewidth=3)
    plt.ylabel('V (mV)')
    plt.grid('on')


error = np.array(error)
#plt.subplot(1,1,2)
plt.figure()
plt.plot(dtArray[1:], error[1:,0], 'k-o', linewidth=3, markersize=10)
plt.grid('on')


'''
plt.subplot(4,1,3)
plt.plot(t, m, 'r', label='m')
plt.plot(t, h, 'g', label='h')
plt.plot(t, n, 'b', label='n')
plt.ylabel('Gating Value')
plt.legend()

plt.subplot(4,1,4)
plt.plot(t, I_inj(t), 'k')
plt.xlabel('t (ms)')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, 31)
'''
plt.show()