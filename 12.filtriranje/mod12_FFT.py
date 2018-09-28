# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:11:38 2018

@author: ASUS
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
plt.rc('font', size = 13, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (10,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
import scipy.fftpack

pod2=np.loadtxt('val2.dat')
pod3=np.loadtxt('val3.dat')
#%%
N = 512
# sample spacing
T = 1.0/512.0
x = np.linspace(0.0, 1.0, N)

plt.figure(0)
plt.grid()
plt.plot(x,pod2+5,color='lightblue', label=r'$val2.dat$')
plt.plot(x,pod3-5,color='darkred', label=r'$val3.dat$')
plt.title(r'$Signala$ $z$ $zamakom$ $\pm 5$')
plt.xlabel(r'$t$')
plt.ylabel(r'$signal$')
plt.legend(loc='best')

#%%
'''osnovni Fourier, brez okenskih funkcij'''
# število točk
N = 512
# vzorčenje
T = 1/N
x = np.linspace(0, 1, N)
yf2 = scipy.fftpack.fft(pod2)
yf3 = scipy.fftpack.fft(pod3)
xf = np.linspace(0, 1/(2*T), N/2)

plt.figure(1)
plt.grid()
plt.plot(xf, 2/N * np.abs(yf3[:N//2])**2,'.',color='darkred', label=r'$val3.dat$')
plt.plot(xf, 2/N * np.abs(yf2[:N//2])**2,'.',color='lightblue', label=r'$val2.dat$')
plt.legend(loc='best')
plt.xlabel(r'$frekvenca$ $\nu$ $[Hz]$')
plt.ylabel(r'$PSD$')
plt.title(r'$frekvenčni$ $spekter$')

#%%
'''Fourier IN okenske funkcije'''
# število točk
N = 512
# vzorčenje
T = 1/N
x = np.linspace(0, 1, N)

'''blackman okno'''
from scipy.signal import blackman
from scipy.fftpack import fft
w = blackman(N)
ywf2 = fft(pod2*w)
yf2 = fft(pod2)
ywf3 = fft(pod3*w)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
plt.figure(2)
plt.plot(xf[1:N//2], 2.0/N * np.abs(yf2[1:N//2]), '-', color='lightblue',label=r'$val3.dat$ $FFT$')
plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf2[1:N//2]), '--', color='lightblue', label=r'$val3.dat$ $FFT$ $in$ $blackman$')
#plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf3[1:N//2]), '-', color='darkred')
#plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf3[1:N//2]), '--', color='darkred')
plt.legend(loc='best')








