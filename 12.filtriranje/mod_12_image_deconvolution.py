# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:25:36 2018

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
plt.rc('figure', figsize = (10,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
import scipy.fftpack
from scipy.fftpack import fft, ifft

####################################################
#a=plt.imread('lincoln_L30_N00.pgm')
#signal0=np.loadtxt('lincoln_L30_N00.pgm',skiprows=3)
signal1=np.loadtxt('signal1.dat')
signal2=np.loadtxt('signal2.dat')
signal3=np.loadtxt('signal3.dat')
import re
import numpy

import numpy as np
import matplotlib.pyplot as plt

def readpgm(name):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    return (np.array(data[3:]),(data[1],data[0]),data[2])
fig = plt.figure(1)
sub = fig.add_subplot(131)
data0 = readpgm('lincoln_L30_N00.pgm')
image0 = np.reshape(data0[0],(256,313))
plt.imshow( image0, cmap='gray')
plt.axis('off')

sub = fig.add_subplot(132)
data1 = readpgm('lincoln_L30_N10.pgm')
image1 = np.reshape(data1[0],data1[1])
plt.imshow( image1, cmap='gray')
plt.axis('off')

sub = fig.add_subplot(133)
data2 = readpgm('lincoln_L30_N20.pgm')
image2 = np.reshape(data2[0],data2[1])
plt.axis('off')
plt.imshow( image2, cmap='gray')
fig.tight_layout()

fig=plt.figure(2)

sub = fig.add_subplot(131)
data3 = readpgm('lincoln_L30_N30.pgm')
image3 = np.reshape(data3[0],data3[1])
plt.imshow( image3, cmap='gray')
plt.axis('off')

sub = fig.add_subplot(132)
data4 = readpgm('lincoln_L30_N40.pgm')
image4 = np.reshape(data4[0],data4[1])
plt.imshow( image4, cmap='gray')
plt.axis('off')
#%%
'''mamo ubistvu matriko 256 x 313, kar pomeni da imas 313 signalov dolgih 256'''
'''spremenljivka t je ubistvu po 256 in predstavlja dolzino...'''

'''primeri signalov'''
plt.figure(3)
stolpec0 = image0[:,0]
stolpec200 = image0[:,200]
stolpec100 = image0[:,100]

T = len(stolpec0)
t = np.linspace(0,T,T)
plt.plot(t,stolpec0,'r--',label='prvi stolpec, brez šuma ')

plt.plot(t,stolpec200,'c--',label='stolpec 100, brez šuma')
plt.plot(t,stolpec100,'b--',label='stolpec 200, brez šuma')

stolpec0 = image3[:,0]
stolpec200 = image3[:,200]
stolpec100 = image3[:,100]

T = len(stolpec0)
t = np.linspace(0,T,T)
plt.plot(t,stolpec0,'r',label='prvi stolpec, z šumom ')
plt.plot(t,stolpec200,'c',label='stolpec 100, z šumom')
plt.plot(t,stolpec100,'b',label='stolpec 200, z šumom')
plt.legend()
plt.grid()
#%%
'''sedaj moramo dekonvoluirati vseh 313 signalov'''

def r(t,tau):
    return 1/tau * np.exp ( -1*t/tau)
tau = 30
rji = r(t,tau)
plt.figure(4)
plt.plot(t,rji,'k+')
plt.grid()
plt.title('Prenosna funkcija r(t)')
'''prvo sliko lahko odpravimo samo z dekonvolucijo ker nima šuma'''
def deconvlution_brezsuma(image,r,t):
    new_image=np.ones(image.shape)
    j=0
    R= fft(r)
    for stolpec in image.T:  
        print(len(stolpec))
        S = fft(stolpec)
        new_stolpec= ifft(S/R)
        new_image[:,j] = new_stolpec
        j +=1
    return new_image
def popravi_sliko(image):
    for i in range(len(image)):
        for j in range(len(image.T)):
            if image[i][j] > 255:
                image[i][j] = 255
            if image[i][j] < 0:
                image[i][j] = 0
    return image
fig = plt.figure(6)
sub = fig.add_subplot(121)
new_image0 = deconvlution_brezsuma(image0,rji,t)
im =plt.imshow(new_image0a,vmin=0, vmax=255,cmap='gray')
plt.axis('off')


sub = fig.add_subplot(122)
new_image1 = deconvlution_brezsuma(image1,rji,t)
im =plt.imshow(new_image1,vmin=0, vmax=255,cmap='gray')
plt.axis('off')


fig = plt.figure(7)
sub = fig.add_subplot(121)
new_image2 = deconvlution_brezsuma(image2,rji,t)
im =plt.imshow(new_image2,vmin=0, vmax=255,cmap='gray')
plt.axis('off')


sub = fig.add_subplot(122)
new_image3 = deconvlution_brezsuma(image3,rji,t)
im =plt.imshow(new_image3,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
    













































