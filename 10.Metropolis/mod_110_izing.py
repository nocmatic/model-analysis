# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:18:53 2017

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
#import scipy.special as spec
#import scipy.stats as stats
#import timeit
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (14,8))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))

def naredi_mrezo(n):    
    generator= np.random.RandomState()
    matrika = np.array([])   
    
    for i in range(n):
        if i == 0 :
            n1 = generator.randint(2,size = n)*2 -1 
            matrika = np.append(matrika, n1)
        else:
            n1 = generator.randint(2,size = n)*2 -1 
            matrika = np.column_stack((matrika, n1))
    #plt.matshow(matrika)
    return matrika
    

def energija(matrika,H,J):
    n = len(matrika)
    E=0
    st_sosedov= 0
    st_stranskih=0
    '''i je vrstica, j je stolpec'''
    for i in range(n):
        for j in range(n):
            dE1 = - H * matrika[i][j]
            E= E + dE1
            if j != n-1:
                dE2 = - J* matrika[i][j]*matrika[i][j+1]                
                E = E + dE2
                st_sosedov +=1
            if i != n-1:
                dE3 = - J* matrika[i][j]*matrika[i+1][j]
                E = E + dE3
                st_sosedov +=1
            if i ==0 : 
                dE4 = - J* matrika[i][j]*matrika[n-1][j]
                E = E + dE4
                st_stranskih +=1
            if j == 0 : 
                dE5 = - J* matrika[i][j]*matrika[i][n-1]
                E = E + dE5
                st_stranskih +=1
    return E
def random_change(matrika,J,H):
    '''naključni element izberemo in mu spremenimo smer, prva naključno število ižreba kateri stolpec,
    drugo pa premik gor ali dol (-1,1)'''
    c = np.array(matrika)
    generator= np.random.RandomState()
    n1 = int(generator.rand()*(len(matrika)))  
    n2=int(generator.rand()*(len(matrika)))  
    
    
    stara = c[n1][n2]
    if stara ==1 : 
        c[n1][n2] = -1
    else:
        c[n1][n2] =1
    nova = c[n1][n2]
    
    k = len(matrika)-1
    #############################################
    deltaE = -H *nova - (-H*stara)
    '''novi sosedje'''
    #print('vrstica:',n1, 'stolpec:',n2)
    if n1 == len(matrika)-1:
        if n2 == len(matrika)-1:
            #print('a')
            Ek = -J* ( nova*c[n1-1][n2] + nova*c[n1][n2-1]+ nova*c[n1][0]+ nova*c[0][n2])
            Ez = -J* ( + stara*c[n1-1][n2] + stara*c[n1][n2-1] + stara*c[n1][0]+ stara*c[0][n2])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
        elif n2 ==0:
            #print('b')
            Ek = -J* ( nova*c[n1-1][n2] + nova*c[n1][n2+1]+ nova*c[0][n2]+ nova*c[n1][k])
            Ez = -J* ( + stara*c[n1-1][n2] + stara*c[n1][n2+1]+ stara*c[0][n2]+ stara*c[n1][k])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
        elif n2 != 0 and n2 != len(matrika)-1: 
            '''sna spodnje  robu '''
            #print('c','vrstica:',n1,'stolpec',n2)
            Ek = -J* (nova * c[n1][n2-1]   + nova*c[n1-1][n2] + nova*c[n1][n2+1]+ nova*c[0][n2])
            Ez = -J* (stara* c[n1][n2-1]  + stara*c[n1-1][n2] + stara*c[n1][n2+1]+ stara*c[0][n2])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
    if n1 ==0:
        if n2 == len(matrika)-1:
            #print('d')
            Ek = -J* ( nova*c[n1+1][n2] + nova*c[n1][n2-1]+  nova*c[k][n2]+ nova*c[n1][0])
            Ez = -J* ( + stara*c[n1+1][n2] + stara*c[n1][n2-1]+ stara*c[k][n2]+ stara*c[n1][0])
            dE = Ek - Ez
            deltaE = deltaE + dE
             
        elif n2 ==0:
            #print('e')
            Ek = -J* ( nova*c[n1+1][n2] + nova*c[n1][n2 +1]+ nova*c[n1][k]+ nova*c[k][n1])
            Ez = -J* ( + stara*c[n1+1][n2] + stara*c[n1][n2+1] + stara*c[n1][k]+ stara*c[k][n1])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
        elif n2 != 0 and n2 !=  len(matrika)-1: 
            '''smo na zgornjem  robu '''
            #print('x')
            Ek = -J* (nova * c[n1][n2-1]   + nova*c[n1+1][n2] + nova*c[n1][n2+1] + nova*c[k][n2])
            Ez = -J* (stara* c[n1][n2-1]  + stara*c[n1+1][n2] + stara*c[n1][n2+1] + stara*c[k][n2])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
            
    if n2 ==0 and n1 != len(matrika)-1 and n1 !=0:
            '''smo na levem robu brez kotov, ki smo jih upoštevali zgoraj'''
            #print('f')
            Ek = -J* (nova * c[n1+1][n2]  + nova*c[n1][n2+1]  + nova*c[n1-1][n2]+ nova*c[n1][k])
            Ez = -J* (stara* c[n1+1][n2]  + stara * c[n1][n2+1] + stara*c[n1-1][n2] + stara*c[n1][k])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
    if n2 == len(matrika)-1 and n1 != len(matrika)-1 and n1 !=0:
            '''smo na desnem robu'''
            #print('g')
            Ek = -J* (nova * c[n1+1][n2]  + nova*c[n1][n2-1]  + nova*c[n1-1][n2]+ nova*c[0][n2])
            Ez = -J* (stara* c[n1+1][n2]  + stara * c[n1][n2-1] + stara*c[n1-1][n2]+  stara*c[0][n2])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
            
    elif n1 !=0 and n2 !=0 and n1 != len(matrika)-1 and n2 != len(matrika)-1:
            #print('h')
            Ek = -J* (nova * c[n1+1][n2]  + nova*c[n1][n2-1]  + nova*c[n1-1][n2] +nova*c[n1][n2+1])
            Ez = -J* (stara* c[n1+1][n2]  + stara * c[n1][n2-1] + stara*c[n1-1][n2] +stara*c[n1][n2+1])
            dE = Ek - Ez
            deltaE = deltaE + dE
            
        
    
    #print('deltaE',deltaE, 'nova-stara',energija(c,1,1)-   energija(matrika,1,1))
    return c, deltaE
def metropolis(zacetno_stanje,n,T0,J,H):
    '''n - število iteracij, T0 = temperatura, alpha = 1 '''
    #T = np.linspace(T0,1,n)
    generator= np.random.RandomState() 
    koncno_stanje= np.array(zacetno_stanje)
    '''prvo stanje'''
    zacetna_energija = energija(zacetno_stanje,J,H)
    koncna_energija = 0
    energije = np.array([])
    
    for i in range(n):
        print (koncna_energija)
        if i ==0:
            koncna_energija = koncna_energija + zacetna_energija
        poskus,deltaE=  random_change(koncno_stanje,J,H)
        if deltaE < 0:
            koncno_stanje= poskus
            koncna_energija = koncna_energija + deltaE
            energije = np.append(energije,koncna_energija)
           
            
            
            continue
        else : 
            n1 = generator.rand()
            boltzman = np.exp(-1*deltaE /((T0)))
            
            if boltzman >= n1:
                koncno_stanje = poskus
                koncna_energija = koncna_energija + deltaE
                
                energije = np.append(energije,koncna_energija)
            else:
                koncno_stanje = koncno_stanje
                energije =np.append(energije,koncna_energija)
                continue
      
    return koncno_stanje, energije, koncna_energija

from matplotlib.colors import ListedColormap
#
#cmap = ListedColormap(['y', 'k'])
#fig = plt.figure(1)
#sub = fig.add_subplot(121)
#a= naredi_mrezo(100)
#sub.imshow(a,cmap)
#plt.title('random')
#
#T0=3 
#J=1
#H=1
#c,E,koncna= metropolis(a,150000,T0,J,H)
#
#sub = fig.add_subplot(132)
#sub.imshow(c,cmap)
#plt.title('urejeno')
#
#sub = fig.add_subplot(133)
#plt.plot(E)
###########################################################################
#def plot_razlicnih_stanj_pri_razlicnih_temperaturah():
#    
#    fig = plt.figure(2)
#    fig2=plt.figure(4)    
#    zacetna_matrika= naredi_mrezo(100)
#    
#    Tmin=0.001
#    Tmax = 4    
#    T = np.linspace(Tmin,Tmax,7)
#    for i in range(len(T)-1):    
#        a= 0
#        if i == 0:
#            print('a')
#            a = 131
#        if i ==1:
#            a = 132
#        if i ==2:
#            a = 133
#        if i == 3:
#            fig = plt.figure(3)
#            fig2= plt.figure(5)
#            a = 131
#        if i == 4:
#            a = 132
#        if i ==5:
#            a = 133
#        sub = fig.add_subplot(a)
#        T0=T[i]
#        J=1
#        H=0
#        c,E,koncna= metropolis(zacetna_matrika,300000,T0,J,H)
#        cmap = ListedColormap(['y', 'k'])        
#        sub.imshow(c,cmap)
#        sub.set_title(r' $\bf{ k_BT = %f }$, n = 300000, H = %d, J = %d' %(T0,H,J))
#       
#        ################################
#        sub = fig2.add_subplot(a)
#        fig2.suptitle('Odvisnost Burn-in od temperature')
#        sub.set_title(r' $\bf{ k_BT = %f }$, n = 300000, H = %d, J = %d' %(T0,H,J))
#        plt.plot(E)
#        plt.grid()
#    fig.tight_layout()
#    fig2.tight_layout()
#    fig2.subplots_adjust(top=0.9)
#    return fig,fig2,T
#fig,fig2,T=plot_razlicnih_stanj_pri_razlicnih_temperaturah()
#################################################################
#
#plt.figure(3)
#plt.plot(E)

#
#
#plt.title('random')
#
#T0=3 
#J=1
#H=0
#c,E,koncna= metropolis(a,1000000,T0,J,H)
#
#sub = fig.add_subplot(122)
#sub.matshow(c)
#plt.title('urejeno')
#
#plt.figure(3)
#plt.plot(E)
####################################################################################################

'''izračun magnetizacij, povprečne enegije, kvadrata magnetizacije...'''
'''PRIREDIL BOM METROPOLIS FUNKCIJO, DA MI NA STANJIH PO BURN INU IZRAČUNA ŽELJENE KOLIČINE IN JIH NATO POVPREČI, 
ZATO ŠE ENKRAT DEFINIRAM
METROPOLIS_poseben(...)'''


def random_change2(c,J,H,energija,zacetna_mag,T0,generator):
    '''naključni element izberemo in mu spremenimo smer, prva naključno število ižreba kateri stolpec,
    drugo pa premik gor ali dol (-1,1)'''
    
    generator= np.random.RandomState()
    n1 = int(generator.rand()*(len(c)))  
    n2=int(generator.rand()*(len(c)))  
    
    
    stara = c[n1][n2]
    if stara ==1 : 
        c[n1][n2] = -1
        S  = zacetna_mag - 2
        
    else:
        c[n1][n2] =1
        S = zacetna_mag +2
        
    nova = c[n1][n2]
    
    k = len(c)-1
    #############################################
    deltaE = -H *nova - (-H*stara)
    '''novi sosedje'''
    #print('vrstica:',n1, 'stolpec:',n2)
    
       
    if deltaE < 0:
            E = energija + deltaE
           
            
            return E,S
            
       
    else : 
            n3 = generator.rand()
            boltzman = np.exp(-1*deltaE /((T0)))
            
            if boltzman >= n3:
                 E = energija + deltaE
               
                 
                 return E,S
            
            else:
            
                c[n1][n2] = stara 
                E = energija 
                 
               
                S = zacetna_mag
                
                return E,S
            
    
    #print('deltaE',deltaE, 'nova-stara',energija(c,1,1)-   energija(matrika,1,1))
    
def metropolis2(zacetno_stanje,n,T0,J,H):
    '''n - število iteracij, T0 = temperatura, alpha = 1 '''
    #T = np.linspace(T0,1,n)
    generator= np.random.RandomState() 
    
    '''prvo stanje'''
    zacetna_energija = energija(zacetno_stanje,J,H)
    zacetna_mag = np.sum(zacetno_stanje)
    koncna_energija = 0
    energije = np.array([])
    #energije_kvadrat =  np.array([])
    mag=  np.array([])
    #mag_kvadrat = np.array([])
    for i in range(n):
        
        print (koncna_energija)
        
        if i ==0:
            
            koncna_energija,S =  random_change2(zacetno_stanje,J,H, zacetna_energija,zacetna_mag,T0,generator)
            nova = zacetno_stanje
            
        else:
            koncna_energija,S =  random_change2(nova,J,H, koncna_energija,S, T0,generator)
            
            if i == n-1:
                return nova,energije,mag,S

            if i%150 ==0:
                energije = np.append(energije,koncna_energija)
                mag = np.append(mag,S)
                #mag_kvadrat = np.append(mag_kvadrat, S_kvadrat)
                #energije_kvadrat =np.append(energije_kvadrat, E_kvadrat)
                
                continue
    return nova, energije,S
                
    
##########################################################
#cmap = ListedColormap(['y', 'k'])
#fig = plt.figure(1)
#sub = fig.add_subplot(121)
#a= naredi_mrezo(100)
#sub.imshow(a,cmap)
#plt.title('random')
#
#T0=3 
#J=1
#H=1
#c,E,koncna= metropolis(a,150000,T0,J,H)
#
#sub = fig.add_subplot(132)
#sub.imshow(c,cmap)
#plt.title('urejeno')
#
#sub = fig.add_subplot(133)
#plt.plot(E)

####################################################


#fig = plt.figure(2)
#sub = fig.add_subplot(121)
#cmap = ListedColormap(['y', 'k'])
#a= naredi_mrezo(100)
#b = np.array(a)
#sub.imshow(a,cmap)
#plt.title('random')
#
#T0=1
#J=1
#H=0
#nova, energije, mag,S= metropolis2(a,1000000,T0,J,H)
#
#sub = fig.add_subplot(132)
#cmap = ListedColormap(['y', 'k'])
#sub.imshow(nova,cmap)
#plt.title('urejeno')
#
#sub = fig.add_subplot(133)
#plt.plot(energije)
#



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

import numba



def ising(N,K,J,H,kb,T):
    #Definicijski del
    #energija=[]
    matrika=np.array([[np.sign(1-2*np.random.rand()) for i in range(N)] for j in range(N)])
    E1=np.sum(-J*(matrika[i%N][j%N]*matrika[(i+1)%N][j%N]+matrika[i%N][j%N]*matrika[i%N][(j+1)%N]+matrika[i%N][j%N]*matrika[(i-1)%N][j%N]+matrika[i%N][j%N]*matrika[i%N][(j-1)%N]) + H*matrika[i][j] for i in range(N) for j in range(N))
   
    
    #algoritmièni del
    for l in range(K):
        #1.korak
        toèki=np.array([np.int(N*np.random.rand()),np.int(N*np.random.rand())])
        
        E2= E1 + 2*J*matrika[toèki[0]][toèki[1]]*(matrika[toèki[0]%N][(toèki[1]+1)%N]+matrika[toèki[0]%N][(toèki[1]-1)%N]+matrika[(toèki[0]+1)%N][(toèki[1])%N]+matrika[(toèki[0]-1)%N][(toèki[1])%N]) + 2*H*matrika[toèki[0]][toèki[1]]
            
        
            
        #2.korak
        
        if np.random.rand() <= np.exp(-(E2-E1)/(kb*T)):
            matrika[toèki[0]][toèki[1]]=-matrika[toèki[0]][toèki[1]]
            E1=E2
        
    #risarski del
    
    return matrika
    
    
    
    #Beep(780, 1000)
    
N = 100
K = 20000
H= 0
J = 1
kb=1
T = 0.00000001
#c,E = ising(N,K,J,H,kb,T)
#test = numba.jit(ising)

#fig = plt.figure(1)
#sub = fig.add_subplot(121)
#cmap = ListedColormap(['y', 'k'])
#sub.imshow(c,cmap)
#sub = fig.add_subplot(122)
#plt.plot(E)
#plt.grid()

#def plot_razlicnih_stanj_pri_razlicnih_temperaturah():
#    
#    fig = plt.figure(2)
#    fig2=plt.figure(4)    
#   
#    
#    Tmin=0.001
#    Tmax = 4    
#    N = 100
#    K = 1000000
#    H= 0
#    J = 1
#    kb=1
#    #T = 1
#
#    T = np.linspace(Tmin,Tmax,6)
#    for i in range(len(T)):    
#        a= 0
#        if i == 0:
#            print('a')
#            a = 131
#        if i ==1:
#            a = 132
#        if i ==2:
#            a = 133
#        if i == 3:
#            fig = plt.figure(3)
#            fig2= plt.figure(5)
#            a = 131
#        if i == 4:
#            a = 132
#        if i ==5:
#            a = 133
#        sub = fig.add_subplot(a)
#        T0=T[i]
#        J=1
#        H=0
#        c,E= ising(N,K,J,H,kb,T0)
#        cmap = ListedColormap(['y', 'k'])        
#        sub.imshow(c,cmap)
#        sub.set_title(r' $\bf{ k_BT = %f }$, n = %e, H = %d, J = %d' %(T0,K,H,J))
#       
#        ################################
#        sub = fig2.add_subplot(a)
#        fig2.suptitle('Odvisnost Burn-in od temperature')
#        sub.set_title(r' $\bf{ k_BT = %f }$, n = %e, H = %d, J = %d' %(T0,K,H,J))
#        plt.plot(E)
#        c
#    fig.tight_layout()
#    fig2.tight_layout()
#    fig2.subplots_adjust(top=0.9)
#    return fig,fig2,T
#fig,fig2,T=plot_razlicnih_stanj_pri_razlicnih_temperaturah()









