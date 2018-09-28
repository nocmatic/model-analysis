# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:52:18 2018

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
signal0=np.loadtxt('signal0.dat')
signal1=np.loadtxt('signal1.dat')
signal2=np.loadtxt('signal2.dat')
signal3=np.loadtxt('signal3.dat')
T= 512
t=  np.linspace(0,T,len(signal0))
'''odkomentiri da narišeš signale'''
fig= plt.figure(16)
plt.xlabel('t [s]')
plt.ylabel('s(t) ')
plt.title('Signal 0 - brez šuma')
plt.grid()
plt.plot(t,signal0, 'b',label='brez šuma')
plt.figure(17)
plt.xlabel('t [s]  ')
plt.title('Signal 2 - rahel šum')
plt.ylabel('s(t)')
plt.grid()
plt.plot(t,signal1,'r', label='Šum 1')
plt.figure(18)
plt.xlabel('t [s]')
plt.ylabel('s(t)')
plt.title('Signal 3 - močnejši šum')
plt.ylabel('s(t)')
plt.grid()
plt.plot(t, signal2, 'g', label = ' Šum 2')
plt.figure(19)
plt.plot(t, signal3, 'c', label ='Šum 3')
plt.xlabel('t [s]')
plt.ylabel('s(t)')
plt.title('Signal 4 - najmočnejši šum')

##################################################################
'''brez šuma lahko naredimo simpl dekonvolucijo'''
def r(t,tau):
    return 0.5 * 1/tau * np.exp ( -1* abs(t)/tau)

tau = 16
t1 = np.linspace(0,256,256)
t2 = np.linspace(-256,-1,256)
'''ni zvezna'''
rji = r(t1,16)
rji = np.append(rji, r(t2,16))

plt.figure(5)
plt.plot(t,rji,'r')
plt.plot(t,rji,'ko',markersize=2.5)
plt.grid()
plt.title('r(t) - periodična')
plt.xlabel('t[s]')
plt.ylabel('r(t)')
'''Vzamemo formulo u(t) = IFFT (S/R) , ker ni šuma...'''
'''poskus na šumu, če ga je ČIst mal še gre......'''
fig = plt.figure(6)
sub = fig.add_subplot(131)

R= fft(rji)
S= fft(signal0)
c = ifft(S/R)

plt.plot(t,c,'r')
plt.xlabel('t[s]')
plt.ylabel('u(t)')
plt.title('Dekonvolucija brez šuma')
plt.grid()

'''poskus na šumu, vidimo da propade zaradi deljenja s 0.'''
sub = fig.add_subplot(132)
R= fft(rji)
S= fft(signal2)
c = ifft(S/R)

plt.plot(t,c,'g')
plt.xlabel('t[s]')
plt.ylabel('u(t)')
plt.title('Dekonvolucija na signalu 2')
plt.grid()



sub = fig.add_subplot(133)
R= fft(rji)
S= fft(signal3)
c = ifft(S/R)

plt.plot(t,c,'b')
plt.xlabel('t[s]')
plt.ylabel('u(t)')
plt.title('Dekonvolucija na signalu 3')
plt.grid()

fig.tight_layout()
#######################################################################

#######################################################################
'''sedaj uporabimo wienerjevo metodo, vzemimo da je da je u(t) = C(w) / R(w) * PHI (w)'''

'''S poznamo , je kar FT od s(t), za N pa mal poskusšamo npr konstantno itd..'''

'''poskus na signalu 2, najprej pogledamo kako izgleda FT od S, da lahko ocenim šum'''
plt.figure(7)
n = len(signal2)
R= fft(rji)
S0= fft(signal0)
S1= fft(signal1)
S2= fft(signal2)
S3= fft(signal3)



freq = 0.5/dt
frekvence = np.linspace(0,freq,int(n/2))
powerSpectrumS0 = abs(S0)*abs(S0)
powerSpectrumS1 = abs(S1)*abs(S1)
powerSpectrumS2= abs(S2)*abs(S2)
powerSpectrumS3 = abs(S3)*abs(S3)
plt.plot(frekvence, np.log(powerSpectrumS0[:int(n/2)]), 'r',label='s0')
plt.plot(frekvence, np.log(powerSpectrumS1[:int(n/2)]), 'g',label='s1')
plt.plot(frekvence, np.log(powerSpectrumS2[:int(n/2)]), 'c',label='s2')
plt.plot(frekvence, np.log(powerSpectrumS3[:int(n/2)]), 'b',label='s3')
plt.grid()
plt.xlabel('frekvenca(Hz)')
plt.ylabel('$log(PSD(f))')
plt.title('Spektralna moč signalov')
plt.legend(loc=0)

N3= np.exp(4.8)
N2 = np.exp(0)
N1 = np.exp(-4.8)
N0 = np.exp(-8)
plt.hlines(4.8,0,0.5,'b')
plt.hlines(0,0,0.5,'c')
plt.hlines(-4.8,0,0.5,'g')
plt.hlines(-8.0,0,0.5,'r')
##############################################################
'''fitat je treba premice ker je signal S(f) = A e^-Bf'''
def fit_premice(f,A,B):
   
    return A*f + B
def vzemi_del(S,mejni,frekvence):
    k=0
    for i in range(len(S)):
        if frekvence[i] <= mejni:
            k +=1
        else:
            break
    return S[:k],k
mejni =np.array([0.13,0.11,0.06,0.055])

S0_meja,k0 = vzemi_del(np.log(powerSpectrumS0),mejni[0],frekvence)
S1_meja,k1 = vzemi_del(np.log(powerSpectrumS1),mejni[1],frekvence)
S2_meja,k2 = vzemi_del(np.log(powerSpectrumS2),mejni[2],frekvence)
S3_meja,k3 = vzemi_del(np.log(powerSpectrumS3), mejni[3],frekvence)
popt0,pcov0 = curve_fit(fit_premice,frekvence[:k0],S0_meja,p0=(1,1))
popt1,pcov1 = curve_fit(fit_premice,frekvence[:k1],S1_meja,p0=(1,1))
popt2,pcov2 = curve_fit(fit_premice,frekvence[:k2],S2_meja,p0=(1,1))
popt3,pcov3 = curve_fit(fit_premice,frekvence[:k3],S3_meja,p0=(1,1))
plt.figure(8)

#
plt.xlabel('frekvenca(Hz)')
plt.ylabel('$log(PSD(f))$')
plt.title('Spektralna moč signalov - fit')
plt.ylim(-40,12)
plt.grid()
plt.plot(frekvence, np.log(powerSpectrumS0[:int(n/2)]), 'r',label='s0')
plt.plot(frekvence, np.log(powerSpectrumS1[:int(n/2)]), 'g',label='s1')
plt.plot(frekvence, np.log(powerSpectrumS2[:int(n/2)]), 'c',label='s2')
plt.plot(frekvence, np.log(powerSpectrumS3[:int(n/2)]), 'b',label='s3')


plt.plot(frekvence, fit_premice(frekvence,*popt0), 'r--',label='s0-fit')
plt.plot(frekvence, fit_premice(frekvence,*popt1), 'g--',label='s1-fit')
plt.plot(frekvence, fit_premice(frekvence,*popt2), 'c--',label='s2-fit')
plt.plot(frekvence, fit_premice(frekvence, *popt3), 'b--',label='s3-fit')


################################################################################################
'''končno izračunamo c(t) -- BLACK MAGIC'''
Skvadrat0 = fit_premice(frekvence,*popt0)
Skvadrat1 = fit_premice(frekvence,*popt1)
Skvadrat2 = fit_premice(frekvence,*popt2)
Skvadrat3 = fit_premice(frekvence, *popt3)

Skvadrat0a=-Skvadrat0 + Skvadrat0[0] + Skvadrat0[-1]
Skvadrat0a = np.linspace(Skvadrat0a[1],Skvadrat0a[-1],256)

Skvadrat1a=-Skvadrat1 + Skvadrat1[0] + Skvadrat1[-1]
Skvadrat1a = np.linspace(Skvadrat1a[0],Skvadrat1a[-2],256)
Skvadrat2a=-Skvadrat2 + Skvadrat2[0] + Skvadrat2[-1]
Skvadrat2a = np.linspace(Skvadrat2a[0],Skvadrat2a[-2],256)
Skvadrat3a=-Skvadrat3 + Skvadrat3[0] + Skvadrat3[-1]
Skvadrat3a = np.linspace(Skvadrat3a[0],Skvadrat3a[-2],256)

Skvadrat0b = np.append(Skvadrat0,Skvadrat0a)
Skvadrat1b = np.append(Skvadrat1, Skvadrat1a)
Skvadrat2b = np.append(Skvadrat2,Skvadrat2a)
Skvadrat3b = np.append(Skvadrat3, Skvadrat3a)

Skvadrat0c = np.exp(Skvadrat0b)
Skvadrat1c = np.exp(Skvadrat1b)
Skvadrat2c = np.exp(Skvadrat2b)
Skvadrat3c = np.exp(Skvadrat3b) 

f3= np.append(frekvence,frekvence[-1] + frekvence,256)
plt.plot(f3, Skvadrat0b,'r--')
plt.plot(f3, Skvadrat1b,'g--')
plt.plot(np.append(frekvence,frekvence[-1] + frekvence), Skvadrat2b,'c--')
plt.plot(np.append(frekvence,frekvence[-1] + frekvence), Skvadrat3b,'b--')

plt.legend()
N3= np.exp(4.8)
N2 = np.exp(0)
N1 = np.exp(-4.8)
N0 = np.exp(-8) 

def phi(S,N):
    return S / (S + N)


def poskusi_N( x,i,j,S,R,S0,k):
    '''x je velikost šuma'''
#    fig=plt.figure(i)
#    sub = fig.add_subplot(k)
#    
    N= x*np.ones(n)
    PHI = phi(S,N)
#    
#    
#    f=np.append(frekvence,frekvence[-1] + frekvence)
#    plt.plot(f,np.log(PHI),'r')
#    plt.plot(f,np.log(N),'g')
#    plt.plot(f,np.log(S),'c')
    #plt.plot(f, PHI,'r')
    fig=plt.figure(i)
    sub = fig.add_subplot(j)
   
    c = ifft(S0/R*PHI)
    plt.plot(t,c,'r')
    plt.xlabel('t[s]')
    plt.ylabel('u(t)')
    plt.title(r'Dekonvolucija signala %d, $log|N(f)|^2$ = %.4f'%(k,np.log(N[0])))
    plt.grid()
    
    return fig
#PHI, c = poskusi_N(N0,9,10,Skvadrat1c,R,S1)
fig = poskusi_N(N1,9,131,Skvadrat1c,R,S1,1)
fig= poskusi_N(N2,9,132,Skvadrat2c,R,S2,2)
fig = poskusi_N(N3,9,133,Skvadrat3c,R,S3,3)
fig.tight_layout()

#######################################################################################
'''ZA KONEC. NA ZADNJI SLIKI VIDIMO DA JE ŠUM ŠE VEDNO PREVELIK ZATO POSKUSIMO UPORABITI VEČJI ŠUM
'''

def poskusi_N2( x,i,j,S,R,S0,k):
    '''x je velikost šuma'''
#    fig=plt.figure(i)
#    sub = fig.add_subplot(k)
#    
    N= x*np.ones(n)
    PHI = phi(S,N)
#    
#    
#    f=np.append(frekvence,frekvence[-1] + frekvence)
#    plt.plot(f,np.log(PHI),'r')
#    plt.plot(f,np.log(N),'g')
#    plt.plot(f,np.log(S),'c')
    #plt.plot(f, PHI,'r')
    fig=plt.figure(i)
    sub = fig.add_subplot(j)
   
    c = ifft(S0/R*PHI)
    plt.plot(t,c,label='$log(|N(f)|^2)$ = %.4f'%(np.log(N[0])))
    plt.xlabel('t[s]')
    plt.ylabel('u(t)')
    plt.title(r'Dekonvolucija signala 3 z različnimi šumi')
    plt.grid()
    

N3= np.exp(4.8)
fig = poskusi_N2(N3,11,111,Skvadrat3c,R,S3,3)


N3= np.exp(6)
fig = poskusi_N2(N3,11,111,Skvadrat3c,R,S3,3)
N3= np.exp(7)
fig = poskusi_N2(N3,11,111,Skvadrat3c,R,S3,3)
N3= np.exp(10)
fig = poskusi_N2(N3,11,111,Skvadrat3c,R,S3,3)
S= fft(signal0)
c = ifft(S/R)
plt.plot(t,c,'k--',label='brez šuma prava deknvolucija')
plt.grid()
plt.legend()