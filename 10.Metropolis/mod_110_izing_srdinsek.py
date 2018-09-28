# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:59:08 2017

@author: matic
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import simplify
from random import random
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import linalg
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from numpy import linalg as LA
from scipy.linalg import solve
from timeit import default_timer as timer
from random import random
from secrets import randbelow
from random import SystemRandom
from scipy.stats import chisquare, norm, kstest, chi2, moment, ks_2samp
from scipy.special import erf
from mpl_toolkits.mplot3d import axes3d
from scipy.signal import gaussian
from winsound import Beep





import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 14))



def isingM(N,K,J,H,kb,T):
    #Definicijski del
    matrika=[[np.sign(1-2*random()) for i in range(N)] for j in range(N)]
    E1=sum(-J*(matrika[i%N][j%N]*matrika[(i+1)%N][j%N]+matrika[i%N][j%N]*matrika[i%N][(j+1)%N]+matrika[i%N][j%N]*matrika[(i-1)%N][j%N]+matrika[i%N][j%N]*matrika[i%N][(j-1)%N]) + H*matrika[i][j] for i in range(N) for j in range(N))
    #algoritmièni del
    mati=[]
    sezi=[]
    sezi2=[]
    eni=[]
    K = 30000000
    k_mejni =20000000
    
    #K=15000
    #k_mejni=1000
    for k in range(K):
        #1.korak
        toèki=[int(N*random()),int(N*random())]
            
        E2= E1 + 2*J*matrika[toèki[0]][toèki[1]]*(matrika[toèki[0]%N][(toèki[1]+1)%N]+matrika[toèki[0]%N][(toèki[1]-1)%N]+matrika[(toèki[0]+1)%N][(toèki[1])%N]+matrika[(toèki[0]-1)%N][(toèki[1])%N]) + 2*H*matrika[toèki[0]][toèki[1]]
        
        if k%10000==0 and k > k_mejni:
            #return matrika
            sezi.append(np.sum(matrika))
            eni.append(E1)

            
        #2.korak
        if random() <= np.exp(-(E2-E1)/(kb*T)):
            matrika[toèki[0]][toèki[1]]=-matrika[toèki[0]][toèki[1]]
            E1=E2
        
    d=len(sezi)
    m=sum(sezi)
    m2=np.sum(sezi[i]*sezi[i] for i in range(d))
        
    n=len(eni)
    s=np.sum(eni)
    s2=np.sum(eni[i]*eni[i] for i in range(n))
    return s/n, (s2/n-s*s/(n*n))/(N*N*kb*T*T), np.abs(m)/d, (m2/d-m*m/(d*d))/(N*N*kb*T) 



def risivse(N,K,J,H,kb,T):
    start=timer()
    tempiè= np.linspace(1,4,50)
    energija=[]
    specifièna=[]
    magnetizacija=[]
    subscetbilnost=[]
    for i in tempiè:
        #mat=isingM(100,1,1,0,1,i)
        #return mat
        E,c,mag,ss=isingM(100,1,1,0,1,i)
        
        energija.append(E)
        specifièna.append(c)
        magnetizacija.append(mag)
        subscetbilnost.append(ss)
    
    plt.figure(0)
    plt.title('Povpreèna energija od temperature J={0} H={1}'.format(J,H))
    plt.plot(tempiè,energija,'b+')
    plt.plot(tempiè,energija,'b--')
    plt.ylabel(r'$\widetilde{E}$')
    plt.xlabel('kT')
    plt.figure(1)
    plt.title('Specifièna toplota od temperature J={0} H={1}'.format(J,H))
    plt.plot(tempiè,specifièna,'r+')
    plt.plot(tempiè,specifièna,'r--')
    plt.ylabel('Specifièna toplota')

    plt.xlabel('kT')
    plt.figure(2)
    plt.title('Magnetizacija od temperature J={0} H={1}'.format(J,H))
    plt.plot(tempiè,magnetizacija,'y+')
    plt.plot(tempiè,magnetizacija,'y--')
    plt.ylabel('Magnetizacija')
    plt.xlabel('kT')

    plt.figure(3)
    plt.title('Spinska subscetibilnost od temperature J={0} H={1}'.format(J,H))
    plt.plot(tempiè,subscetbilnost,'g+')
    plt.plot(tempiè,subscetbilnost,'g--')
    plt.ylabel('Magnetna suscetibilnost')
    plt.xlabel('kT')
    end=timer()
    print(end-start)
    Beep(780, 1000)
    return tempiè,energija,specifièna,magnetizacija,subscetbilnost

N=1
K = 1
J=1
H=0
kb=1
T=40
#a= risivse(N,K,J,H,kb,T)
#tempiè,energija1,specifièna,magnetizacija,subscetbilnost= risivse(N,K,J,H,kb,T)
tempiè = np.delete(tempiè,(5,7,8))
magnetizacija = np.delete(magnetizacija,(5,7,8))
specifièna = np.delete(specifièna,(5,7,8))
subscetbilnost =np.delete(subscetbilnost,(5,7,8))
plt.figure(0)
plt.title('Povpreča energija (T) J={0} H={1}'.format(J,H))
#plt.plot(tempiè,energija,'b+')
#plt.plot(tempiè,energija,'b--')
plt.ylabel(r'$\widetilde{E}$')
plt.xlabel('kT')
plt.figure(1)
plt.title('Specifièna toplota (T) J={0} H={1}'.format(J,H))
plt.plot(tempiè,specifièna,'r+')
plt.plot(tempiè,specifièna,'r--')
plt.ylabel('Specifièna toplota')
plt.grid()

plt.xlabel('kT')
plt.figure(2)
plt.title('Magnetizacija (T) J={0} H={1}'.format(J,H))
plt.plot(tempiè,magnetizacija,'y+')
plt.plot(tempiè,magnetizacija,'y--')
plt.ylabel('Magnetizacija')
plt.xlabel('kT')
plt.grid()
plt.figure(3)
plt.title('Magnetna subscetibilnost (T) J={0} H={1}'.format(J,H))
plt.plot(tempiè,subscetbilnost,'g+')
plt.plot(tempiè,subscetbilnost,'g--')
plt.ylabel('Magnetna suscetibilnost')
plt.xlabel('kT')
plt.grid()
