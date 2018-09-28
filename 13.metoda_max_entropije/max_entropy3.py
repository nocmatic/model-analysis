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
import time
pi = np.pi

from scipy import linalg
from scipy import optimize




f2 = open("co2.dat")


d2 = np.loadtxt(f2)
x = d2[86::, 0]
y = d2[86::, 1]



def f(x, a, b):
    return a*x + b
params, errs = optimize.curve_fit(f, x, y , p0=(1, 300))
a, b =  params
y = [y[k] - f(x[k], a, b) for k in range(len(y))]
plt.figure(0)
plt.plot(d2[:,0],d2[:,1],'b--',label='$co2.dat$')
plt.plot(x,f(x,*params),'r',label='linearni fit')
plt.title('$co2.dat$')
plt.grid()
plt.xlabel('leto')
plt.ylabel('koncentracija $CO_2$ [ppm]')
plt.legend()

fig = plt.figure(5)
plt.plot(x, y)
plt.xlabel('leto')
plt.ylabel('koncentracija $CO_2$ [ppm]')
plt.title("Koncentracija $CO_2$ z odštetim linearni naraščanjem")
plt.show()
plt.grid() 
fig.savefig("co1.pdf")

sig = y

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
            roots[j] = 1/np.conj(roots[j])
    fixed_coefs = np.polynomial.polynomial.polyfromroots(roots)
    return roots, np.real(fixed_coefs)


def uber_func(sig, p):
    a = coefs(sig, p)
    z, a = fix_coefs(a)
    return z, a




def circ(r, phi):
    return r*np.cos(phi), r*np.sin(phi)

phis = np.linspace(0, 2*pi, 100)

fig = plt.figure(1,figsize=(8,8))
for p in [5, 10, 25, 35, 50]:
    z, a = uber_func(sig, p)
    Rez = np.real(z)
    Imz = np.imag(z)
    plt.plot(Rez, Imz, "o", label='p=%s'%str(p))
plt.legend(loc="best", )
plt.plot(*circ(1, phis), "k")
plt.grid(True)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Poli za co2.dat")
plt.show()
fig.savefig("poles2.pdf")

def PSD(spekter):
	"""Iz spektra izracuna spekter moci."""
	PSD = []
	prvi=spekter[0]
	zadnji=spekter[-1]
	PSD.append(prvi*prvi)
	
	for i in range(1, (len(spekter)-1)//2):
		PSD.append(0.5*(np.absolute(spekter[i])**2 + np.absolute(spekter[-(i+1)])**2))
	PSD.append(zadnji*zadnji)
	najvec=max(PSD)
	PSD=PSD

	return PSD
omegas = np.linspace(0, pi, 1000)
fig=plt.figure(2)
for p in [5, 10, 25, 35]:
    z, a = uber_func(sig, p)
    plt.plot(omegas, [P(w, a) for w in omegas],label='p=%s'%str(p))

plt.grid(True)
fft_sig = fft(sig[13:])
plt.plot(np.linspace(0,pi,len(sig)/2+1),np.array(PSD(fft(sig))),'.',label="fft")
plt.yscale("log")
plt.xlabel(r"$\omega$")
plt.ylabel(r"$logPSD(\omega)$")
plt.title("Frekvečni spekter za $co2.dat$")
plt.legend(loc="best", )
plt.show()


########################################################################
#%%
def S(a,p,signal):
    s = signal[len(signal)-p:]
    print(len(s))
    vsota = a*np.flip(s,axis=0)
    vsota = np.sum(vsota)
    return vsota

sig = y
z, a = uber_func(sig, p)
def napoved(a,signal_y,signal_x,p,stevilo_iteracij,st_let):
    new_y = np.zeros(stevilo_iteracij)
    new_x =  np.ones(stevilo_iteracij)
    x =np.linspace(0,st_let,stevilo_iteracij)
    k=1
    new_x= x
    for i in range(stevilo_iteracij):
        
        new_y_clen = S(a,len(a),signal_y)
        signal_y = np.append(signal_y, new_y_clen)
        

  
    return np.append(signal_x,new_x + signal_x[-1]),signal_y
plt.figure(10)
new_x,new_y = napoved(a,sig,x,20,13,1)

plt.plot( new_x, new_y,'b')
    
