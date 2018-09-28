# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:21:39 2018

@author: matic
"""

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
plt.rc('font', size = 15, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (17,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
import scipy.fftpack
from scipy.fftpack import fft, ifft
import time

import numpy as np
import matplotlib.pyplot as plt
import time
pi = np.pi
from numpy import fft
from scipy import linalg
from scipy import optimize

f2 = open("borza.dat")
d2 = np.loadtxt(f2)
#x = d2[86::, 0]
#y = d2[86::, 1]

y=d2
x = np.arange(len(y))
#%%





fig = plt.figure(1)
plt.plot(x,y,'-',color='darkred',label='$borza.dat$')
plt.title('$borza.dat$')
plt.grid()
plt.xlabel('t')
plt.ylabel(r'cena')
plt.legend(loc='best')
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
def uber_func1(sig, p):
    a = coefs(sig, p)
    z=0
    #z, a = fix_coefs(a)
    return z, a




def circ(r, phi):
    return r*np.cos(phi), r*np.sin(phi)
#%%
#phis = np.linspace(0, 2*pi, 100)
##
#fig = plt.figure(2,figsize=(8,8))
#for p in [5, 10, 30, 60]:
#    z, a = uber_func(sig, p)
#    Rez = np.real(z)
#    Imz = np.imag(z)
#    plt.plot(Rez, Imz, "o", label='p=%s'%str(p))
#plt.legend(loc="best", )
#plt.plot(*circ(1, phis), "-")
#plt.grid(True)
#plt.xlim(-1.1, 1.1)
#plt.ylim(-1.1, 1.1)
#plt.title("poli $borza.dat$")
#plt.show()
##fig.savefig("poles2.pdf")
#
#
#omegas = np.linspace(0, pi, 1000)
#fig=plt.figure(3)
#for p in [5, 15, 25, 40]:
#    z, a = uber_func(sig, p)
#    plt.plot(omegas/pi, [P(w, a) for w in omegas],label='p=%s'%str(p))
#plt.legend(loc="best", title=r"$p$")
#plt.grid(True)
#plt.yscale("log")
#plt.xlabel(r"$\omega$")
#plt.ylabel(r"$\log(PSD(\omega))$")
#plt.title("$frekvenčni$ $spekter$ $borza.dat$")
#plt.show()
#fig.savefig("freq_co1.pdf")

#%%


def S(a,p,signal):
    s = signal[len(signal)-p:]
    vsota = a*np.flip(s,axis=0)
    vsota = -1* np.sum(vsota)
#    print(vsota)
    return vsota



def napoved(a,signal_y,signal_x,p,stevilo_iteracij,st_let):
#    new_y = np.zeros(stevilo_iteracij)
    #new_x =  np.ones(stevilo_iteracij)
    x = np.linspace(0,st_let,stevilo_iteracij)
    new_x = x
    print(new_x)
    for i in range(stevilo_iteracij):
        new_y_clen = S(a,len(a),signal_y)
        print(new_y_clen)
        signal_y = np.append(signal_y, new_y_clen)
    signal_y = signal_y[int(len(y)/2):]
    signal_x = new_x + signal_x[-1]
    return signal_x,signal_y

f2 = open("borza.dat")
d2 = np.loadtxt(f2)
#x = d2[86::, 0]
#y = d2[86::, 1]

y0=d2
x0 = np.arange(len(y))
sig = y0[:int(len(y0)/2)]
x = x0[:int(len(x0)/2)]


#p=10
z, a = uber_func(sig, p)
new_x1,new_y1 = napoved(a,sig,x,p,600,300)
p=25
z, a = uber_func(sig, p)
new_x2,new_y2 = napoved(a,sig,x,p,350,350)




plt.figure(5)
#plt.plot(new_x1, new_y1,'b',label='napoved p = 10')
plt.plot(new_x2, new_y2,'g',label='napoved p = 20')
#plt.plot(new_x3, new_y3,'k',label='napoved p = 35 ')
#plt.plot(new_x4 ,new_y4,'y',label='napoved p = 25 ')
plt.plot(x0, y0, color='darkred',alpha = 0.5,label='$opazovanja$')
plt.ylim(-2000,10000)
plt.grid()
plt.title('$borza.dat$')
plt.grid()
plt.xlabel('t')
plt.ylabel(r'cena')
plt.legend(loc='best')
