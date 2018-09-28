# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:13:41 2017

@author: matic
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import optimize
from scipy.optimize import curve_fit
from numpy.linalg import solve
from scipy.optimize import minimize
from cycler import cycler
import scipy.special as spec
import scipy.stats as stats
import timeit
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (14,8))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
generator= np.random.RandomState()
def nakljucni_sprehod_mejava_smeri_en_nevtron(d,l):
    R=0
    T=0
    N=0
    i = 0
    x=0
    while(True):
            if (x > d):
                T = T+1
                N = N+1
                break
            if (x < 0):
                R=R+1
                N = N+1
                break
            else:
                if (i%2 == 0):
                    n1 = generator.rand()
                    s = -l * np.log(1-n1)
                    x = x + s
                    i=i+1
                    print(x, 'iteracija %d'%i)
                else: 
                    n1 = generator.rand()
                    s = -l * np.log(1-n1)
                    x = x -s
                    i= i+1
                    print(x, 'iteracija %d'%i)
    return R,T,N,i
     
def nakljucni_sprehod_menjava_smeri_simulacija(n,d,l):
    '''l - povprečna prosta pot, R = radij krogle'''    
    x = 0
    R=0
    T=0
    N=0  
    stevilo_sprehodov= np.array([])
    for i in range(n):
        R1,T1,N1,j=nakljucni_sprehod_mejava_smeri_en_nevtron(d,l)
        R=R + R1
        T = T+T1
        N = N+N1
        stevilo_sprehodov=np.append(stevilo_sprehodov, j)
        print(R,T,N)

    R =  R/N
    T = T/N
    povp_stevilo_sprehodov = stevilo_sprehodov.mean()
    err = stevilo_sprehodov.std()
    return R,T,N,povp_stevilo_sprehodov,err
    
#d = 1
#l=0.2
#n=100
#R,T,N = nakljucni_sprehod_menjava_smeri_simulacija(n,d,l)           
#            

def nakljucni_sprehod_žrebanje_smeri_en_nevtron(d,l):
    R=0
    T=0
    N=0
    i = 0
    x=0
    while(True):
            n1 = generator.rand()
            s = -l * np.log(1-n1)
            n2 = np.random.randint(2)
            if (i==0):
                x = x + s
                i=i+1
                print(x, 'iteracija %d'%i)
                continue
            if n2 ==1:
                        x = x + s
                        i=i+1
                        print(x, 'iteracija %d'%i)
                       
                        
            elif n2==0:
                        x = x - s
                        i=i+1
                        print(x, 'iteracija %d'%i)
                       
            if (x > d):
                T = T+1
                N = N+1
                break
            if (x < 0):
                R=R+1
                N = N+1
                break                   
                                                      
    return R,T,N,i
     
def nakljucni_sprehod_žrebanje_smeri_simulacija(n,d,l):
    '''l - povprečna prosta pot, R = radij krogle'''    
    x = 0
    R=0
    T=0
    N=0
    stevilo_sprehodov= np.array([])
    for i in range(n):
        R1,T1,N1,j=nakljucni_sprehod_žrebanje_smeri_en_nevtron(d,l)
        R=R + R1
        T = T+T1
        N = N+N1
        stevilo_sprehodov=np.append(stevilo_sprehodov, j)
        print(R,T,N)
    
    R =  R/N
    T = T/N
    povp_stevilo_sprehodov = stevilo_sprehodov.mean()
    err = stevilo_sprehodov.std()
    return R,T,N,povp_stevilo_sprehodov,err
#d = 1
#l=0.05
#n=10000
#R2,T2,N2= nakljucni_sprehod_žrebanje_smeri_simulacija(n,d,l)           
#                    
def plot_menjava_smeri():
    plt.figure(1)
    plt.xlabel('q')
    plt.ylabel('R/T [%]')
    plt.title(r'Prepustnost za n = 10000 v odvisnosti od $q = \frac{l}{d}$')
    
    d = 1
    l = np.linspace(0.01,1.5,50)
    x = l/d
    n = 10000
    R = np.array([])
    T = np.array([])
    stevilo_sprehodov = np.array([])
    st_err= np.array([])
    for i in l:
        R1,T1,N1,st,err = nakljucni_sprehod_menjava_smeri_simulacija(n,d,i)
        R = np.append(R,R1)
        T = np.append(T,T1)
        stevilo_sprehodov = np.append(stevilo_sprehodov,st)
        st_err= np.append(st_err,err)
    
    plt.plot(x, R, 'r+', label = 'R')
    plt.plot(x, T, 'b+', label ='T')
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.xlabel('q')
    plt.ylabel('R/T [%]')
    plt.title(r'Povprečno število sprehodov dvisnosti od $q = \frac{l}{d} $')
    plt.plot(x, stevilo_sprehodov, 'r+', label = 'Povprečno število sprehodov - Model A (menjava smeri)')
    plt.errorbar(x,stevilo_sprehodov,st_err,fmt='o')
    plt.legend()
    return R,T,stevilo_sprehodov,st_err
    
#R,T= plot_menjava_smeri()
def plot_nakljucne():
    plt.figure(3)
    plt.xlabel('q')
    plt.ylabel('R/T [%]')
    plt.title(r'Prepustnost (naključne smeri) za n = 10000 v odvisnosti od $q = \frac{l}{d}$')
    
    d = 1
    l = np.linspace(0.01,1.5,50)
    x = l/d
    n = 10000
    R = np.array([])
    T = np.array([])
    stevilo_sprehodov = np.array([])
    st_err= np.array([])
    for i in l:
        R1,T1,N1,st,err = nakljucni_sprehod_žrebanje_smeri_simulacija(n,d,i)
        R = np.append(R,R1)
        T = np.append(T,T1)
        stevilo_sprehodov = np.append(stevilo_sprehodov,st)
        st_err= np.append(st_err,err)
    
    plt.plot(x, R, 'r+', label = 'R')
    plt.plot(x, T, 'b+', label ='T')
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.xlabel('q')
    plt.ylabel('R/T [%]')
    plt.title(r'Menjava smeri - povprečno število sprehodov dvisnosti od $q = \frac{l}{d} $')
    plt.plot(x, stevilo_sprehodov, 'b+', label = 'Povprečno število sprehodov - Model B (naključni žreb smeri)')
    plt.legend()
    
    plt.errorbar(x,stevilo_sprehodov,st_err,fmt='o',color='black')
    plt.grid()
    return R,T,stevilo_sprehodov,st_err
#R1,T1,st1,err1 = plot_nakljucne()
#    
#R2,T2,st2,err2 = plot_menjava_smeri()
 
#d = 1
#l = np.linspace(0.001,2,100)
#x = l/d
#plt.figure(2)
#plt.xlabel('q')
#plt.ylabel(r'$log \widetilde{N}$')
#plt.title(r'Menjava smeri - povprečno število sprehodov dvisnosti od $q = \frac{l}{d} $')
#plt.plot(x, np.log(st1), 'b+', label = ' Model B (naključni žreb odboja)')
#plt.legend()
#plt.errorbar(x,np.log(st1), np.log(err1),fmt='bo',color='black')
#plt.grid()
#
#
#plt.figure(2)
#plt.xlabel('q')
#plt.ylabel(r'$log \widetilde{N}$')
#plt.title(r'Povprečno število sprehodov dvisnosti od $q = \frac{l}{d} $')
#plt.plot(x,np.log( st2), 'r+', label = ' Model A (menjava smeri odboja)')
#plt.errorbar(x,np.log(st2),np.log(err2),fmt='ro')
#plt.legend()