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
#plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
plt.rc('axes', prop_cycle=(cycler('color', ['k', 'r','g','b','y', 'm', 'k'])))
import scipy.fftpack
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import time
pi = np.pi
from numpy import fft
from scipy import linalg
from scipy import optimize

from scipy.fftpack import fft

pod2=np.loadtxt('val2.dat')
pod3=np.loadtxt('val3.dat')


    plt.legend(loc='best')
    plt.grid()
    plt.xlabel(r'frekvenca ')
    plt.ylabel(r'PSD')
    plt.title('val%s.dat,'%(st_val))


f2 = open("val2.dat")
f3 = open("val3.dat")

d2 = np.loadtxt(f2)
d3 = np.loadtxt(f3)
pi = np.pi


sig2 = d2
sig = d3
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

print(np.exp(1j))

p = 5
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
            print('i')
            roots[j] = 1/np.conj(roots[j])
    fixed_coefs = np.polynomial.polynomial.polyfromroots(roots)
    return roots, np.real(fixed_coefs)

def fix_coefs_withoutfix(a):
    m = len(a)
    a = np.append([1], a)
    roots = np.roots(a)
    
   
    return roots, np.real(a)

def uber_func(sig, p):
    a = coefs(sig, p)
    z, a = fix_coefs(a)
    return z, a

def uber_func_without(sig, p):
    a = coefs(sig, p)
    z, a = fix_coefs_withoutfix(a)
    return z, a



def circ(r, phi):
    return r*np.cos(phi), r*np.sin(phi)

phis = np.linspace(0, 2*pi, 100)

fig = plt.figure(2, figsize=(8,8))
for p in [10,20,30,40]:
    z, a = uber_func_without(sig, p)
    Rez = np.real(z)
    Imz = np.imag(z)
    plt.plot(Rez, Imz, "o", label=str(p))
plt.legend(loc="best", title=r"$p$")
plt.plot(*circ(1, phis), "k")
plt.grid(True)

plt.title("Poli brez preslikave za val3.dat")
plt.show()


phis = np.linspace(0, 2*pi, 100)

fig = plt.figure(3, figsize=(8,8))
for p in [ 10,20,30,40]:
    z, a = uber_func(sig, p)
    Rez = np.real(z)
    Imz = np.imag(z)
    plt.plot(Rez, Imz, "o", label=str(p))
plt.legend(loc="best", title=r"$p$")
plt.plot(*circ(1, phis), "k")
plt.grid(True)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Poli s preslikavo za val3.dat")
plt.show()
fig.savefig("poles1.pdf")

fig = plt.figure(4, figsize=(8,8))
for p in [5,8,10,15]:
    z, a = uber_func_without(sig, p)
    Rez = np.real(z)
    Imz = np.imag(z)
    plt.plot(Rez, Imz, "o", label=str(p))
plt.legend(loc="best", title=r"$p$")
plt.plot(*circ(1, phis), "k")
plt.grid(True)

plt.title("Manj polov brez preslikave za val3.dat")
plt.show()


phis = np.linspace(0, 2*pi, 100)

fig = plt.figure(5, figsize=(8,8))
for p in [5,8,10,15]:
    z, a = uber_func(sig, p)
    Rez = np.real(z)
    Imz = np.imag(z)
    plt.plot(Rez, Imz, "o", label=str(p))
plt.legend(loc="best", title=r"$p$")
plt.plot(*circ(1, phis), "k")
plt.grid(True)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Manj polov s preslikavo za val3.dat")
plt.show()
fig.savefig("poles1.pdf")


omegas = np.linspace(0, pi, 1000)
fig=plt.figure(7)
for p in [5, 8, 14, 20]:
    z, a = uber_func(sig, p)
    plt.semilogy(omegas*250/pi, [(P(w, a)) for w in omegas], label='p = %s'%str(p))
plt.legend(loc="best",)

plt.xlabel(r"$\omega$")
plt.ylabel(r"$log(PSD(\omega))$")
plt.title("Frekvečni spekter za $val3.dat$")
plt.show()
plt.grid()
fig.savefig("freq2.pdf")



omegas = np.linspace(0, pi, 1000)
fig=plt.figure(9)
for p in [5,  8, 14, 20]:
    z, a = uber_func(sig2, p)
    plt.semilogy(omegas*250/pi, [(P(w, a)) for w in omegas], label='p = %s'%str(p))
plt.legend(loc="best",)

plt.xlabel(r"$\omega$")
plt.ylabel(r"$log(PSD(\omega))$")
plt.title("Frekvečni spekter za $val2.dat$")
plt.grid()
plt.show()
fig.savefig("freq2.pdf")

