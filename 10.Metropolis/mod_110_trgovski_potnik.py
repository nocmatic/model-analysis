# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:40:57 2017

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
import numba
from numba import double, jit,int64
@numba.jit
def energija(tocke,L):
    E0=0
    '''zacetna energija'''
    for i in range(len(tocke)):
        E1 = np.sqrt( (tocke[i%L][0]- tocke[(i+1)%L][0]) * (tocke[i%L][0]- tocke[(i+1)%L][0])  + 
                      (tocke[i%L][1]- tocke[(i+1)%L][1])*  (tocke[i%L][1]- tocke[(i+1)%L][1]))
        E0 +=E1
    return E0
@numba.jit
def zacetni_polozaj(N,L):
    tocke=  np.array([0,0])
    L1=0
    for k in range(0,N,L):
        for j in range(0,N,L):
            if j ==0 and k==0:
                continue;
            tocke = np.vstack((tocke,np.array([k,j])))
           
            L1=L1+1
    
    L = len(tocke)
#        L_zacetni.append([0,0])
#        L_zacetni.append([1,0])
#        L_zacetni.append([0,1])
#        L1= L1+1
#        break
            
    return tocke
#    for i in range(1,N):
#        L_zacetni = np.vstack((L_zacetni,np.array([j[i],l[i]])))
#    return L_zacetni
#    while (len(L_zacetni) > L):
#        
#    tocka= int(N*random())
    
    

#    

def trgovski_potnik(N,L,st_iteracij,T,zacetni):
    '''N je velikost mreže'''
    '''M je število mest'''
    #Definicijski del - ustvarimo mrežo za kasneje in nato izžrebamo pozicijo mest

    
    '''izbero M točk za začetno pozicijo mest'''
    
    '''enakomerno, dodaj za naključno...'''
    
    #tocke =zacetni_polozaj(N,L)
    tocke = zacetni
    
    L = len(tocke)
    
    #E0 = energija(tocke,L)
    E0=0
    '''zacetna energija'''
    for i in range(len(tocke)):
        E1 =  (tocke[i%L][0]- tocke[(i+1)%L][0]) * (tocke[i%L][0]- tocke[(i+1)%L][0]) + (tocke[i%L][1]- tocke[(i+1)%L][1])*  (tocke[i%L][1]- tocke[(i+1)%L][1])
        E0 +=E1
    K = st_iteracij
    #energije=np.zeros(int(K/100000))
    for k in range(K):
       
        #1.korak
        i = int(L*np.random.rand())
        j = int(L*np.random.rand())

#        
        e1b = (tocke[(i)%L][0]-tocke[(i+1)%L][0])*(tocke[(i)%L][0]-tocke[(i+1)%L][0])+ (tocke[(i)%L][1]-tocke[(i+1)%L][1])*(tocke[(i)%L][1]-tocke[(i+1)%L][1]) 
        e2b=  (tocke[(i)%L][0]-tocke[(i-1)%L][0])*(tocke[(i)%L][0]-tocke[(i-1)%L][0])+ (tocke[(i)%L][1]-tocke[(i-1)%L][1])*(tocke[(i)%L][1]-tocke[(i-1)%L][1]) 
        e3b=  (tocke[(j)%L][0]-tocke[(j-1)%L][0])*(tocke[(j)%L][0]-tocke[(j-1)%L][0])+ (tocke[(j)%L][1]-tocke[(j-1)%L][1])*(tocke[(j)%L][1]-tocke[(j-1)%L][1]) 
        e4b=  (tocke[(j)%L][0]-tocke[(j+1)%L][0])*(tocke[(j)%L][0]-tocke[(j+1)%L][0])+ (tocke[(j)%L][1]-tocke[(j+1)%L][1])*(tocke[(j)%L][1]-tocke[(j+1)%L][1]) 
        
        nova = tocke[(j)%L].copy()
        stara = tocke[(i)%L].copy()
        tocke[i%L] = nova
        tocke[(j)%L]= stara
        
        e1 = (tocke[(i)%L][0]-tocke[(i+1)%L][0])*(tocke[(i)%L][0]-tocke[(i+1)%L][0])+ (tocke[(i)%L][1]-tocke[(i+1)%L][1])*(tocke[(i)%L][1]-tocke[(i+1)%L][1]) 
        e2=  (tocke[(i)%L][0]-tocke[(i-1)%L][0])*(tocke[(i)%L][0]-tocke[(i-1)%L][0])+ (tocke[(i)%L][1]-tocke[(i-1)%L][1])*(tocke[(i)%L][1]-tocke[(i-1)%L][1])
        e3= (tocke[(j)%L][0]-tocke[(j-1)%L][0])*(tocke[(j)%L][0]-tocke[(j-1)%L][0])+ (tocke[(j)%L][1]-tocke[(j-1)%L][1])*(tocke[(j)%L][1]-tocke[(j-1)%L][1]) 
        e4=  (tocke[(j)%L][0]-tocke[(j+1)%L][0])*(tocke[(j)%L][0]-tocke[(j+1)%L][0])+ (tocke[(j)%L][1]-tocke[(j+1)%L][1])*(tocke[(j)%L][1]-tocke[(j+1)%L][1]) 
         
        E2 = E0 +e1 +e2 +e3+e4 - e1b-e2b-e3b-e4b            
        #2.korak        
        
        a =np.exp(-(E2-E0)/(T))
        
        #print('boltzman:',a)
           
        if (np.random.rand()<= a):
            
            E0 = E2
            #if k %1000000 ==0:
                #print('boltzman:',a)
            # print('iteracija:',k,'energija:',E2) 
            continue
        tocke[i%L] = stara
        tocke[(j)%L]= nova
    print('boltzman:',a)
        
    return tocke


def crte(resitev,L):
    zac = tocke[0]
    kon = tocke[-1]
    print(zac,kon)
    plt.plot(tocke[:,0],tocke[:,1],alpha=0.5)
    plt.scatter(zac[0],zac[1],c='r',label='zacetek',)
    plt.scatter(kon[0],kon[1],c='g',label='konec',)
    plt.legend()
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
N = 100
L =5
aniling = 500
st_iteracij = 10000000
T=5
zacetni=zacetni_polozaj(N,L) 
test = jit(trgovski_potnik)

T = np.linspace(10000,0.001,aniling)
start=timer()
#for i in range(len(T)):
#    if i ==0:
#        tocke1 = test(N,L,st_iteracij,T[i],zacetni)
#    else: 
#        tocke1 = test(N,L,st_iteracij,T[i],tocke1)
#
#



E = energija(tocke1,len(tocke1))
end =timer()
print (end-start)
#energija(tocke,L1)
#stare,tocke,energije,L1 = trgovski_potnik(N,L,st_iteracij,temp,stare,L1)
def plot(tocke,st_iteracij):
    fig=plt.figure(11)
    sub = fig.add_subplot(121)
    # plt.plot(energije)
    plt.grid()
    plt.title('Samplanje nekaj energij')
    plt.xlabel('Sample')
    plt.ylabel('Energija')
    sub = fig.add_subplot(122)
    fig.suptitle('Simulirano ohlajanje za N = %d mest, k = %e, stopenj ohlajanja = %d'%(len(tocke),aniling*st_iteracij,aniling))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.scatter(tocke[:,0],tocke[:,1])
    plt.grid()

    zac = tocke[0]
    kon = tocke[-1]
    tocke  = np.vstack((tocke,zac))
    
   
    plt.plot(tocke[:,0],tocke[:,1],alpha=0.5)
    plt.scatter(zac[0],zac[1],c='k',label='zacetek',)
    plt.scatter(kon[0],kon[1],c='g',label='konec',)
    plt.legend()
  
plot(tocke1,st_iteracij)