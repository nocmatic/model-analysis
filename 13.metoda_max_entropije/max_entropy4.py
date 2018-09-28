import numpy as np
import matplotlib.pyplot as plt
import time
pi = np.pi
from numpy import fft
from scipy import linalg
from scipy import optimize


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import optimize
from scipy.optimize import curve_fit
#from numpy.linalg import solve
#from scipy.optimize import minimize
from cycler import cycler
plt.rc('text', usetex = False)
plt.rc('font', size = 20, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (17,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
plt.rc('axes', prop_cycle=(cycler('color', ['k', 'r','g','darkred','y', 'purple', 'k'])))
import scipy.fftpack
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import time


def R_func(s, p):
    N = len(s)
    dmy_Rs = [sum([s[j]*s[j+i] for j in range(N-i)])/(N-i) for i in range(p)]
    return dmy_Rs

def rightR_func(s, p):
    N = len(s)
    dmy_Rs = [-sum([s[j]*s[j+i] for j in range(N-i)])/(N-i) for i in range(1, p+1)]
    return dmy_Rs

def miniR(s, k):
    N = len(s)
    return sum([s[j]*s[j+k]/(N-k) for j in range(N-k)])

def P(w, a):
    return np.abs(sum([a[k]*np.exp(-1j*w*k) for k in range(p+1)]))**(-2)/512**2



def coefs(sig, p):
    R = R_func(sig, p)
    toeplitzR = linalg.toeplitz(R)
    rightR = rightR_func(sig, p)

    a = linalg.solve(toeplitzR, rightR)
    return a

def fix_coefs(a):
    m = len(a)
    a = np.append([1], a)
    roots = np.roots(a)
    for j in range(len(roots)):
        if np.abs(roots[j]) > 1:
            roots[j] = 1/np.conj(roots[j])
    fixed_coefs = np.polynomial.polynomial.polyfromroots(roots)
    return roots, np.real(fixed_coefs)


def uber_func(sig, p):
    a = coefs(sig, p)
    z, a = fix_coefs(a)
    return z, a

p = 7
ts = np.linspace(0, 100,1000)

dws = np.logspace(-0.5, -2, 5)
print(dws)
def f(t, dw):
    return np.sin(2*pi*t) + np.sin((2*pi + 2*pi*dw)*t)
plt.figure(0)
t = ts
plt.plot(t,f(t,dws[0]), 'r',label=r'$\Delta \omega = %f$'%dws[0])
plt.plot(t,f(t,dws[2]), 'g',label=r'$\Delta \omega = %f$'%dws[2])
plt.plot(t,f(t,dws[4]), 'b', label=r'$\Delta \omega = %f$'%dws[4])
plt.title(r' $s(t) = sin(2\pi t) + sin(2\pi t + \Delta \omega)$, dt = 0.1')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.grid()
plt.legend()
omegas = np.linspace(0, pi, 1000)
fig=plt.figure()
dt = 1000/10000
ny = 0.5/dt
for dw in dws:
    sig = [f(t, dw) for t in ts]
    z, a = uber_func(sig, p)
    plt.plot(omegas*ny/pi, [P(w, a) for w in omegas], label=str(round(dw, 4)))
plt.legend(loc="best", title=r"$\Delta \omega$")
plt.grid(True)
plt.yscale("log")
plt.xlabel(r"$\nu$")
plt.ylabel(r"$logPSD(\nu)$")
plt.title("Frekvečni spekter, $dt = 0.1, p =7$ ")
plt.show()
fig.savefig("freq_2peak.pdf")

