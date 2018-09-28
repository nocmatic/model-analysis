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
plt.rc('font', size = 14, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (17,7))
plt.rc('lines', linewidth=2.0)
#plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
plt.rc('axes', prop_cycle=(cycler('color', ['lightgreen', 'y','lightblue','darkred','c', 'm', 'k'])))
import scipy.fftpack
from scipy.fftpack import fft, ifft
#%%
####################################################
#a=plt.imread('lincoln_L30_N00.pgm')
#signal0=np.loadtxt('lincoln_L30_N00.pgm',skiprows=3)
#signal1=np.loadtxt('signal1.dat')
#signal2=np.loadtxt('signal2.dat')
#signal3=np.loadtxt('signal3.dat')
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

sub = fig.add_subplot(121)
data3 = readpgm('lincoln_L30_N30.pgm')
image3 = np.reshape(data3[0],data3[1])
plt.imshow( image3, cmap='gray')
plt.axis('off')

sub = fig.add_subplot(122)
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
plt.plot(t,stolpec0,'-',color='g',label='prvi stolpec, brez šuma ')

plt.plot(t,stolpec200,'-',color='r',label='stolpec 100, brez šuma')
plt.plot(t,stolpec100,'-',color='k',label='stolpec 200, brez šuma')

stolpec0 = image3[:,0]
stolpec200 = image3[:,200]
stolpec100 = image3[:,100]

T = len(stolpec0)
t = np.linspace(0,T,T)
plt.plot(t,stolpec0,'g--',label='prvi stolpec, z šumom ')
plt.plot(t,stolpec200,'r--',label='stolpec 100, z šumom')
plt.plot(t,stolpec100,'k--',label='stolpec 200, z šumom')
plt.legend(loc='best')
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
im =plt.imshow(new_image0,vmin=0, vmax=255,cmap='gray')
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
#%%    
''' računamo povprečje stolpcev '''
povp0 = np.mean(image0, axis=1)
povp1 = np.mean(image1, axis=1)
povp2 = np.mean(image2, axis=1)
povp3 = np.mean(image3, axis=1)
povp4 = np.mean(image4, axis=1)

#plt.figure(10)
fft1 = fft(povp1)
powerSpectrumF1 = abs(fft1)*abs(fft1)
fft2 = fft(povp2)
powerSpectrumF2 = abs(fft2)*abs(fft2)
fft3 = fft(povp3)
powerSpectrumF3 = abs(fft3)*abs(fft3)
fft4 = fft(povp4)
powerSpectrumF4 = abs(fft4)*abs(fft4)

plt.figure(9)
plt.plot(np.log(powerSpectrumF1), 'r', label='slika 1')
plt.plot(np.log(powerSpectrumF2),'g', label='slika 2')
plt.plot(np.log(powerSpectrumF3), 'b', label='slika 3')
plt.plot(np.log(powerSpectrumF4), 'k',label='slika 4')
plt.hlines(7.6,0,256,color='r')
plt.hlines(7.4,0,256,color='g')
plt.hlines(7.1,0,256,color='b')
plt.hlines(6.5,0,256,color='k')
plt.ylim(5,15)
plt.grid()
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel(r'$\log{PSD}$')
plt.title('Spektralna moč')
plt.legend(loc=0)
mejna_sum = np.array([7.6, 7.4, 7.1, 6.5]) #odčitane vrednosti, ''taprave''
#mejna_sum = np.array([7.6, 8.4, 9.1, 12.5])#poskušam da vidm če so kej boljše slike
mejna_st = [70, 60, 50, 45]
frekvence = np.linspace(0,255,256)

def fit_eksp(f,A,B,C):
    return A*np.exp(-B*f) + C

def fit_lin(f,D,E):
    return D*f + E

#popt1,pcov1 = curve_fit(fit_eksp,frekvence[1:50],np.log(powerSpectrumF1)[1:50],p0=(1,1,10))
popt1lin,pcov1lin = curve_fit(fit_lin,frekvence[1:70],np.log(powerSpectrumF1)[1:70],p0=(-1,10))
popt1lin2,pcov1lin2 = curve_fit(fit_lin,frekvence[187:255],np.log(powerSpectrumF1)[187:255],p0=(1,2.1))
popt2lin,pcov12lin = curve_fit(fit_lin,frekvence[1:60],np.log(powerSpectrumF2)[1:60],p0=(-1,10))
popt2lin2,pcov2lin2 = curve_fit(fit_lin,frekvence[197:255],np.log(powerSpectrumF2)[197:255],p0=(1,2.1))
popt3lin,pcov3lin = curve_fit(fit_lin,frekvence[1:50],np.log(powerSpectrumF3)[1:50],p0=(-1,10))
popt3lin2,pcov3lin2 = curve_fit(fit_lin,frekvence[207:255],np.log(powerSpectrumF3)[207:255],p0=(1,-2.1))
popt4lin,pcov4lin = curve_fit(fit_lin,frekvence[1:45],np.log(powerSpectrumF4)[1:45],p0=(-1,20))
popt4lin2,pcov4lin2 = curve_fit(fit_lin,frekvence[212:255],np.log(powerSpectrumF4)[212:255],p0=(3,-4))

fig = plt.figure(10)
sub1 = fig.add_subplot(121)
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel(r'$\log{PSD}$')
plt.title('linearni fit')
#plt.ylim(-40,12)
plt.grid()
plt.plot(frekvence,np.log(powerSpectrumF1), color='r',label='$slika$ 1')
plt.plot(frekvence,np.log(powerSpectrumF2), color='b',label='$slika$ 2')
#plt.plot(frekvence[1:50],fit_eksp(frekvence[1:50],*popt1), '--',color='orange',label='$slika$ $1$ - $fit$')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt1lin), '--',color='r')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt1lin2), '--',color='r')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt2lin), '--',color='b')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt2lin2), '--',color='b')
plt.legend(loc='best')

sub2 = fig.add_subplot(122)
plt.grid()
plt.plot(frekvence,np.log(powerSpectrumF3), color='k',label='$slika$ 3')
plt.plot(frekvence,np.log(powerSpectrumF4), color='g',label='$slika$ 4')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt3lin), '--',color='k')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt3lin2), '--',color='k')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt4lin), '--',color='g')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt4lin2), '--',color='g')
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel(r'$\log{PSD}$')
plt.title('linearni fit')
#plt.ylim(-40,12)
plt.legend(loc='best')

Skvadrat1 = fit_lin(frekvence[:128],*popt1lin)
Skvadrat2 = fit_lin(frekvence[:128],*popt2lin)
Skvadrat3 = fit_lin(frekvence[:128],*popt3lin)
Skvadrat4 = fit_lin(frekvence[:128],*popt4lin)
Skvadrat1a = fit_lin(frekvence[128:256],*popt1lin2)
Skvadrat2a = fit_lin(frekvence[128:256],*popt2lin2)
Skvadrat3a = fit_lin(frekvence[128:256],*popt3lin2)
Skvadrat4a = fit_lin(frekvence[128:256],*popt4lin2)

Skvadrat1b = np.append(Skvadrat1,Skvadrat1a)
Skvadrat2b = np.append(Skvadrat2, Skvadrat2a)
Skvadrat3b = np.append(Skvadrat3,Skvadrat3a)
Skvadrat4b = np.append(Skvadrat4, Skvadrat4a)

Skvadrat1c = np.exp(Skvadrat1b)
Skvadrat2c = np.exp(Skvadrat2b)
Skvadrat3c = np.exp(Skvadrat3b)
Skvadrat4c = np.exp(Skvadrat4b) 

Ncrta1 = np.exp(mejna_sum[0]*np.ones(256))
Ncrta2 = np.exp(mejna_sum[1]*np.ones(256))
Ncrta3 = np.exp(mejna_sum[2]*np.ones(256))
Ncrta4 = np.exp(mejna_sum[3]*np.ones(256))

def phi(S1,N1):
    return S1/(S1+N1)

def deconvlution_s_sumom(image,r,Skvadrat,N,t):
    new_image=np.ones(image.shape)
    j=0
    R= fft(r)
    for stolpec in image.T:  
        print(len(stolpec))
        S = fft(stolpec)
        PHI = phi(Skvadrat,N)
        new_stolpec= ifft(S/R*PHI)
        new_image[:,j] = new_stolpec
        j +=1
    return new_image

def r(t,tau):
    return 1/tau * np.exp ( -1*t/tau)
tau = 30
rji = r(t,tau)
#
stolpec0 = image3[:,0]
T = len(stolpec0)
t = np.linspace(0,T,T)

fig = plt.figure(11)

sub = fig.add_subplot(131)
new_image1 = deconvlution_s_sumom(image2,rji,Skvadrat2c,Ncrta2,t)
im =plt.imshow(new_image2,vmin=0, vmax=255,cmap='gray')
plt.axis('off')


sub = fig.add_subplot(132)
new_image3 = deconvlution_s_sumom(image3,rji,Skvadrat3c,Ncrta3,t)
im =plt.imshow(new_image1,vmin=0, vmax=255,cmap='gray')
plt.axis('off')

sub = fig.add_subplot(133)
new_image4 = deconvlution_s_sumom(image4,rji,Skvadrat4c,Ncrta4,t)
im =plt.imshow(new_image2,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.suptitle('Linearni fit')

mejna_sum = np.array([7.6, 7.4, 7.1, 6.5]) #odčitane vrednosti, ''taprave''
#mejna_sum = np.array([7.6, 8.4, 9.1, 12.5])#poskušam da vidm če so kej boljše slike
mejna_st = [70, 60, 50, 45]
#########################################################################################
'''proba z cutoff'''
fig=plt.figure(13)
def cut_off(mejna_st,image):
    first = image[:mejna_st]
    second = image[255-mejna_st:256]
    
    new_image = np.append(first,-30*np.ones(255-2*mejna_st) )
    new_image = np.append(new_image,second)
    print(len(new_image))
    return new_image
plt.figure(19)

cutoff=mejna_st[1]

new_fft1 = cut_off(cutoff,np.log(powerSpectrumF1))
plt.plot(frekvence,new_fft1,'b')
plt.title('cut off')
plt.grid()

Skvadrat1c = np.exp(new_fft1)
fig = plt.figure(14)
sub = fig.add_subplot(131)
Ncrta1 = np.exp(8*np.ones(256))
new_image1 = deconvlution_s_sumom(image1,rji,Skvadrat1c,Ncrta1,t)
im =plt.imshow(new_image1,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 1 cut_off: %d, šum: %d'%(cutoff,np.log(Ncrta1[0])))



sub = fig.add_subplot(132)
cutoff= mejna_st[2]
new_fft2 = cut_off(cutoff,np.log(powerSpectrumF2))
#plt.plot(frekvence,new_fft1,'b')



Skvadrat2c = np.exp(new_fft2)

Ncrta2 = np.exp(9*np.ones(256))
new_image2 = deconvlution_s_sumom(image2,rji,Skvadrat2c,Ncrta2,t)
im =plt.imshow(new_image2,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 2 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta2[0])))

sub = fig.add_subplot(133)
cutoff = mejna_st[3]
new_fft3 = cut_off(cutoff,np.log(powerSpectrumF3))
#plt.plot(frekvence,new_fft1,'b')


Skvadrat3c = np.exp(new_fft3)

Ncrta3 = np.exp(8*np.ones(256))
new_image3 = deconvlution_s_sumom(image3,rji,Skvadrat3c,Ncrta3,t)
im =plt.imshow(new_image3,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 3 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta3[0])))

################################################################################################
'''primerjava cutoff na zadnji'''



def cut_off(mejna_st,image):
    first = image[:mejna_st]
    second = image[255-mejna_st:256]
    
    new_image = np.append(first,np.zeros(255-2*mejna_st))
    new_image = np.append(new_image,second)
    print(len(new_image))
    return new_image
plt.figure(19)

cutoff=mejna_st[1]

new_fft1 = cut_off(cutoff,np.log(powerSpectrumF1))
plt.plot(frekvence,new_fft1,'b')
plt.title('cut off')
plt.grid()

Skvadrat1c = np.exp(new_fft1)
fig = plt.figure(18)










sub = fig.add_subplot(131)
cutoff = 80
new_fft3 = cut_off(cutoff,np.log(powerSpectrumF1))
#plt.plot(frekvence,new_fft1,'b')


Skvadrat3c = np.exp(new_fft3)

Ncrta3 = np.exp(10*np.ones(256))
new_image3 = deconvlution_s_sumom(image1,rji,Skvadrat3c,Ncrta3,t)
im =plt.imshow(new_image3,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 2 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta3[0])))


sub = fig.add_subplot(132)
cutoff = 60
new_fft3 = cut_off(cutoff,np.log(powerSpectrumF1))
#plt.plot(frekvence,new_fft1,'b')


Skvadrat3c = np.exp(new_fft3)

Ncrta3 = np.exp(10*np.ones(256))
new_image3 = deconvlution_s_sumom(image1,rji,Skvadrat3c,Ncrta3,t)
im =plt.imshow(new_image3,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 2 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta3[0])))


sub = fig.add_subplot(133)
cutoff = 30
new_fft3 = cut_off(cutoff,np.log(powerSpectrumF1))
#plt.plot(frekvence,new_fft1,'b')


Skvadrat3c = np.exp(new_fft3)

Ncrta3 = np.exp(10*np.ones(256))
new_image3 = deconvlution_s_sumom(image1,rji,Skvadrat3c,Ncrta3,t)

from scipy import ndimage
from scipy import misc
face = misc.face(gray=True).astype(float)



filter_blurred_f = ndimage.gaussian_filter(new_image3, 1)
alpha = 1
sharpened = new_image3 + alpha * (new_image3- filter_blurred_f)
im =plt.imshow(sharpened,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 2 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta3[0])))


##################################################################
'''poskus fit eksponenta'''

#%%
mejna_st = [70, 60, 50, 45]
frekvence = np.linspace(0,255,256)

def fit_lin(f,A,B,C):
    return A*np.exp(-B*f) +C


def fit_lin(f,A,B,C):
    return A*np.sqrt(f) + B*f  + C
p0=[1,1,1]
p01=[1,1,1]
#popt1,pcov1 = curve_fit(fit_eksp,frekvence[1:50],np.log(powerSpectrumF1)[1:50],p0=(1,1,10))
#popt1lin,pcov1lin = curve_fit(fit_lin,frekvence[1:70],np.log(powerSpectrumF1)[1:70],p0)
#popt1lin2,pcov1lin2 = curve_fit(fit_lin,frekvence[187:255],np.log(powerSpectrumF1)[187:255],p01)
#popt2lin,pcov12lin = curve_fit(fit_lin,frekvence[1:60],np.log(powerSpectrumF2)[1:60],p0)
#popt2lin2,pcov2lin2 = curve_fit(fit_lin,frekvence[197:255],np.log(powerSpectrumF2)[197:255],p01)
#popt3lin,pcov3lin = curve_fit(fit_lin,frekvence[1:50],np.log(powerSpectrumF3)[1:50],p0)
#popt3lin2,pcov3lin2 = curve_fit(fit_lin,frekvence[207:255],np.log(powerSpectrumF3)[207:255],p01)
#popt4lin,pcov4lin = curve_fit(fit_lin,frekvence[1:45],np.log(powerSpectrumF4)[1:45],p0)
#popt4lin2,pcov4lin2 = curve_fit(fit_lin,frekvence[212:255],np.log(powerSpectrumF4)[212:255],p01)
#


popt1lin,pcov1lin = curve_fit(fit_lin,frekvence[1:70],np.log(powerSpectrumF1)[1:70],p0)
popt1lin2,pcov1lin2 = curve_fit(fit_lin,frekvence[187:255],np.log(powerSpectrumF1)[187:255],p01)
popt2lin,pcov12lin = curve_fit(fit_lin,frekvence[1:60],np.log(powerSpectrumF2)[1:60],p0)
popt2lin2,pcov2lin2 = curve_fit(fit_lin,frekvence[197:255],np.log(powerSpectrumF2)[197:255],p01)
popt3lin,pcov3lin = curve_fit(fit_lin,frekvence[1:50],np.log(powerSpectrumF3)[1:50],p0)
popt3lin2,pcov3lin2 = curve_fit(fit_lin,frekvence[207:255],np.log(powerSpectrumF3)[207:255],p01)
popt4lin,pcov4lin = curve_fit(fit_lin,frekvence[1:45],np.log(powerSpectrumF4)[1:45],p0)
popt4lin2,pcov4lin2 = curve_fit(fit_lin,frekvence[212:255],np.log(powerSpectrumF4)[212:255],p01)

fig = plt.figure(20)
sub1 = fig.add_subplot(121)
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel(r'$\log{PSD}$')
plt.title('eksponentni fit')
#plt.ylim(-40,12)
plt.grid()
plt.plot(frekvence,np.log(powerSpectrumF1), color='r',label='$slika$ 1')
plt.plot(frekvence,np.log(powerSpectrumF2), color='b',label='$slika$ 2')
#plt.plot(frekvence[1:50],fit_eksp(frekvence[1:50],*popt1), '--',color='orange',label='$slika$ $1$ - $fit$')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt1lin), '--',color='r')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt1lin2), '--',color='r')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt2lin), '--',color='b')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt2lin2), '--',color='b')
plt.legend(loc='best')

sub2 = fig.add_subplot(122)
plt.grid()
plt.plot(frekvence,np.log(powerSpectrumF3), color='k',label='$slika$ 3')
plt.plot(frekvence,np.log(powerSpectrumF4), color='g',label='$slika$ 4')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt3lin), '--',color='k')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt3lin2), '--',color='k')
plt.plot(frekvence[:128],fit_lin(frekvence[:128],*popt4lin), '--',color='g')
plt.plot(frekvence[128:255],fit_lin(frekvence[128:255],*popt4lin2), '--',color='g')
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel(r'$\log{PSD}$')
plt.title('eksponentni fit')
#plt.ylim(-40,12)
plt.legend(loc='best')

Skvadrat1 = fit_lin(frekvence[:128],*popt1lin)
Skvadrat2 = fit_lin(frekvence[:128],*popt2lin)
Skvadrat3 = fit_lin(frekvence[:128],*popt3lin)
Skvadrat4 = fit_lin(frekvence[:128],*popt4lin)
Skvadrat1a = fit_lin(frekvence[128:256],*popt1lin2)
Skvadrat2a = fit_lin(frekvence[128:256],*popt2lin2)
Skvadrat3a = fit_lin(frekvence[128:256],*popt3lin2)
Skvadrat4a = fit_lin(frekvence[128:256],*popt4lin2)

Skvadrat1b = np.append(Skvadrat1,Skvadrat1a)
Skvadrat2b = np.append(Skvadrat2, Skvadrat2a)
Skvadrat3b = np.append(Skvadrat3,Skvadrat3a)
Skvadrat4b = np.append(Skvadrat4, Skvadrat4a)

Skvadrat1c = np.exp(Skvadrat1b)
Skvadrat2c = np.exp(Skvadrat2b)
Skvadrat3c = np.exp(Skvadrat3b)
Skvadrat4c = np.exp(Skvadrat4b) 

Ncrta1 = np.exp(mejna_sum[0]*np.ones(256))
Ncrta2 = np.exp(mejna_sum[1]*np.ones(256))
Ncrta3 = np.exp(mejna_sum[2]*np.ones(256))
Ncrta4 = np.exp(mejna_sum[3]*np.ones(256))


fig=plt.figure(13)
def cut_off(mejna_st,image):
    first = image[:mejna_st]
    second = image[255-mejna_st:256]
    
    new_image = np.append(first,-30*np.ones(255-2*mejna_st) )
    new_image = np.append(new_image,second)
    print(len(new_image))
    return new_image
plt.figure(19)

cutoff=50
new_fft1 = cut_off(cutoff,Skvadrat1b)
plt.plot(frekvence,new_fft1,'b')
plt.title('cut off')
plt.grid()

Skvadrat1c = np.exp(new_fft1)
fig = plt.figure(25)
sub = fig.add_subplot(131)
Ncrta1 = np.exp(9*np.ones(256))
new_image1 = deconvlution_s_sumom(image1,rji,Skvadrat1c,Ncrta1,t)
im =plt.imshow(new_image1,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 2 cut_off: %d, šum: %d'%(cutoff,np.log(Ncrta1[0])))



sub = fig.add_subplot(132)
cutoff= 50
new_fft2 = cut_off(cutoff,Skvadrat1b)
#plt.plot(frekvence,new_fft1,'b')



Skvadrat2c = np.exp(new_fft2)

Ncrta2 = np.exp(9*np.ones(256))
new_image2 = deconvlution_s_sumom(image2,rji,Skvadrat2c,Ncrta2,t)
im =plt.imshow(new_image2,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 3 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta2[0])))

sub = fig.add_subplot(133)
cutoff = 50
new_fft3 = cut_off(cutoff,Skvadrat1b)
#plt.plot(frekvence,new_fft1,'b')


Skvadrat3c = np.exp(new_fft3)

Ncrta3 = np.exp(10.5*np.ones(256))
new_image3 = deconvlution_s_sumom(image3,rji,Skvadrat3c,Ncrta3,t)
im =plt.imshow(new_image3,vmin=0, vmax=255,cmap='gray')
plt.axis('off')
plt.title('Slika 4 cut_off: %d, šum: %d'%(cutoff, np.log(Ncrta3[0])))
plt.suptitle('Poskus eksponetnega fita')
