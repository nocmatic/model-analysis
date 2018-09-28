# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:18:21 2017

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

'''prvi del - KONSTANTNA GOSTOTA'''
generator= np.random.RandomState()

def volumen_valji2(n,R):
    x1 = generator.rand(n)*R
    y1 = generator.rand(n)*R
    z1 = generator.rand(n)*R
    m = 0
    N = 0
    comparex = x1**2 + y1**2 <=R**2  
    comparey = y1**2 + z1**2 <=R**2 
    comparez = x1**2 + z1**2 <=R**2
    
    compare = np.logical_and(comparex,comparey)
    comparefinal = np.logical_and(compare,comparez)
    
    
    m= np.sum(comparefinal)
    
    return m/n*8*R**3
def theta_solve(x,u):
        return u - 1/float(4) * ( 3*np.cos(x) - (np.cos(x))**3) - 1/float(2)

def nakljucni_gama_krogla(n,R,l):
    '''l - povprečna prosta pot, R = radij krogle'''
    n1 = generator.rand(n)
    n2 = generator.rand(n)
    n3= generator.rand(n)
    '''izračunaj thete in r'''
    costheta = 2*n1-1
    
    theta = np.arccos( costheta )
    r = R * np.cbrt( n2)
    d = -l * np.log(1-n3)
    
    
    t = -1 * r * np.cos(theta) + np.sqrt(R*R - r*r*np.sin(theta)*np.sin(theta))
   
    compare = d-t >= 0
    m=np.sum(compare)
    
    return m / n
def porazdelitev_tetiv(n,R):
    n1 = generator.rand(n)
    n2= generator.rand(n)
    '''izračunaj thete in r'''
    costheta = 2*n1-1
    
    theta = np.arccos( costheta )
    r = np.cbrt(  R* n2)
    t = -1 * r * np.cos(theta) + np.sqrt(R*R - r*r*np.sin(theta)*np.sin(theta))
    fig =plt.figure(5)
    plt.xlabel('tetive t')
    plt.ylabel(r'$\frac{dP}{dt}$')
    plt.hist(t,bins=50,normed=True)
    plt.grid()
    plt.figure(6)
    plt.grid()
    plt.xlabel('r')
    plt.ylabel(r'$\frac{dP}{dr}$')
    plt.plot(np.linspace(0,1,100),3*(np.linspace(0,1,100))**2,'r',label = r'$ \frac{dP}{dr} = 3r^2$')
    plt.legend()
    plt.hist(r,bins=50, normed=True)
    plt.figure(7)
    plt.grid()
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\frac{dP}{d\theta}$')
    plt.plot(np.linspace(0,np.pi,100),0.5*np.sin((np.linspace(0,np.pi,100))),'r',label = r'$\frac{dP}{d\theta}= 0.5 sin(\theta)$')
    plt.hist(theta,bins=50, normed=True)
    
    plt.legend()
    return t,r
#R= 1
#l = 1
#n = 100000
#a = nakljucni_gama_krogla(n,R,l)
#    
#        
#R= 1
#l =10
#n = 100000
#b= nakljucni_gama_krogla(n,R,l)
#
#l = np.linspace(0.001,10,50)
#R = 1
#n= 10000000
#
#n= 10000000
#prepustno = np.array([])
#for i in l:
#    a = nakljucni_gama_krogla(n,R,i)
#    prepustno = np.append(prepustno,a)
#plt.figure(2)    
#plt.plot(l/R,prepustno,'r+',label='Prepustnost (%)')
#plt.xlabel(r'$ \frac{\widetilde{l}}{R} $')
#
#plt.grid()
#plt.title('Prepustnost in absorbcija v odvinosti od razmerja radija in povprečne proste poti')
#
#plt.plot(l/R,1-prepustno,'b+', label= 'Absorbcija (%)')
#plt.legend()
#plt.ylabel('%')
#
#l =1
#R=1
#n= 1000000
#b= nakljucni_gama_krogla(n,R,l)

#def volumen_valji2(n,R):
#    x1 = generator.rand(n)*R
#    y1 = generator.rand(n)*R
#    z1 = generator.rand(n)*R
#    m = 0
#    rho0 = 1
#    comparex = x1**2 + y1**2 <=R**2  
#    comparey = y1**2 + z1**2 <=R**2 
#    comparez = x1**2 + z1**2 <=R**2
#    compare = np.logical_and(comparex,comparey)
#    comparefinal = np.logical_and(compare,comparez)
#    mji = np.where(comparefinal, comparefinal, 0)*1
#    mji2 = np.where(comparefinal, comparefinal, 0)*1**2
#    m = np.sum(mji)
#    m2 = np.sum(mji2)
#    V = m / n *8
#    
#    var_enega_zreba = (m2/n)- (m/n)**2
#    sigma_V  = V * var_enega_zreba / np.sqrt(m)
#    
#    gostota0 = np.where(comparefinal, comparefinal, 0)*rho0    
#    gostota1 = np.sum(gostota0)
#    povp_gostota = gostota1/m
#    masa = povp_gostota*V    
#    gostota2 = np.where(comparefinal, comparefinal, 0)*rho0**2
#    povp_gostota2 = np.sum(gostota2)/m
#    var_gostota = (gostota2 - povp_gostota**2)
#    sigma_masa = V * var_gostota / np.sqrt(m)
#    
#    
#    return var_V
#    vztrajn = np.where(comparefinal, comparefinal, 0)*rho0*(x1**2+y1**2+z1**2)
#    J = np.sum(vztrajn)/n*8
#    var_J = np.sum(vztrajn**2)/n*8 - J**2
#    
#    return V, masa, J, var_V, var_masa, var_J

###########################################################################################
'''porazdelitev tetiv'''
#t,r= porazdelitev_tetiv(n,1)

def sigma(st_poskusov, n,l,R):
    x = np.array([])
    for i in range(st_poskusov):
        x = np.append(x, nakljucni_gama_krogla(n,R,l))
    povp = x.mean()
    
    std = np.sqrt(np.mean(abs(x - povp)**2))
    return povp,std,x
st_pos= 1000
n=100
l=1
R=1
povp, std ,x= sigma(st_pos,n,l,R)
print(povp,std)

n=10000
l=1
R=1
povp, std,x = sigma(st_pos,n,l,R)
print(povp,std)

#n=100000
#l=1
#R=1
#povp, std,x = sigma(st_pos,n,l,R)
#print(povp,std)
def koren(n,a,b):
    return a + b/np.sqrt(n)
def plotaj_napake():
#    nji = np.linspace(100,100000,100)
#    sigme = np.array([])
#    l=1
#    R=1
#    for i in nji:
#        povp, std,x = sigma(100,int(i),l,R)
#        sigme = np.append(sigme, std)
##        
    plt.figure(9)
    plt.xlabel(r'$N$ ')
    plt.ylabel(r'$\sigma$')
    plt.grid()
    
    n= np.power(nji,-1/float(2))
    plt.plot(nji,sigme,'b+')
    popt,pcov = curve_fit(koren,nji,sigme,p0=(0,-2.402530733520421e-05))
    print (popt)
    plt.plot(nji, koren(nji,popt[0],popt[1]),'r',label=r'fit $\sigma = a + \frac{b}{N^{1/2}}$')
    plt.legend()
    plt.text(40000,0.02,'a,b = %e,%e'%(popt[0],popt[1]))
    
    plt.figure(10)
    plt.plot(nji**(-0.5),sigme, 'b+')
    return nji,sigme,n


nji,sigme,n= plotaj_napake()
   

