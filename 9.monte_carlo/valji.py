# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:42:43 2017

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

'''prvi del - KONSTANTNA GOSTOTA'''
generator= np.random.RandomState()

def volumen_valji(n,R):
    x1 = generator.rand(n)*R
    y1 = generator.rand(n)*R
    z1 = generator.rand(n)*R
    m = 0
    N = 0
    for i in range(n):
        if (x1[i]**2 + y1[i]**2 < R**2) and (y1[i]**2 + z1[i]**2 < R**2) and (x1[i]**2 + z1[i]**2 < R**2):
            m = m+1
            N = N+1
        else: 
            m = m 
            N = N+1
    return m/N

def volumen_valji2(n,R):
    x1 = generator.rand(n)*R
    y1 = generator.rand(n)*R
    z1 = generator.rand(n)*R
    m = 0
    rho0 = 1
    comparex = x1**2 + y1**2 <=R**2  
    comparey = y1**2 + z1**2 <=R**2 
    comparez = x1**2 + z1**2 <=R**2
    compare = np.logical_and(comparex,comparey)
    comparefinal = np.logical_and(compare,comparez)
    mji = np.where(comparefinal, comparefinal, 0)*1
    mji2 = np.where(comparefinal, comparefinal, 0)*1**2
    m = np.sum(mji)
    m2 = np.sum(mji2)
    V = m / n *8
    
    var_enega_zreba = (m2/n)- (m/n)**2
    sigma_V  = V * var_enega_zreba / np.sqrt(m)
    
    gostota0 = np.where(comparefinal, comparefinal, 0)*rho0    
    gostota1 = np.sum(gostota0)
    povp_gostota = gostota1/m
    masa = povp_gostota*V    
    
    gostota2 = np.where(comparefinal, comparefinal, 0)*rho0**2
    povp_gostota2 = np.sum(gostota2)/m
    
    var_gostota = (povp_gostota2 - povp_gostota**2)
    sigma_masa = V * var_gostota / np.sqrt(m)
    
    vztrajn0 = np.where(comparefinal, comparefinal, 0)*rho0*(x1**2+y1**2+z1**2)
    vztrajn1 = np.where(comparefinal, comparefinal, 0)*rho0**2*(x1**2+y1**2+z1**2)**2
    vztrajn2 = np.sum(vztrajn1)
    J = np.sum(vztrajn0)/m
    var_J = (vztrajn2/m) - J**2
    sigma_J = V * var_J / np.sqrt(m)
    
    return V, masa, J, sigma_V, sigma_masa, sigma_J 

def volumen_valji3(n,R,p):
    x1 = generator.rand(n)*R
    y1 = generator.rand(n)*R
    z1 = generator.rand(n)*R
    m = 0
    rho1 = (np.sqrt(x1**2 + y1**2 + z1**2)/R)**p
    comparex = x1**2 + y1**2 <=R**2  
    comparey = y1**2 + z1**2 <=R**2 
    comparez = x1**2 + z1**2 <=R**2
    compare = np.logical_and(comparex,comparey)
    comparefinal = np.logical_and(compare,comparez)
    mji = np.where(comparefinal, comparefinal, 0)*1
    mji2 = np.where(comparefinal, comparefinal, 0)*1**2
    m = np.sum(mji)
    m2 = np.sum(mji2)
    V = m / n *8
    
    var_enega_zreba = (m2/n)- (m/n)**2
    
    sigma_V  = V * var_enega_zreba / np.sqrt(m)
    
    gostota0 = np.where(comparefinal, comparefinal, 0)*rho1   
    gostota1 = np.sum(gostota0)
    povp_gostota = gostota1/m
    masa = povp_gostota*V    
    gostota2 = np.where(comparefinal, comparefinal, 0)*rho1**2
    povp_gostota2 = np.sum(gostota2)/m
    var_gostota = (povp_gostota2 - povp_gostota**2)
    sigma_masa = V * var_gostota / np.sqrt(m)
    
    vztrajn0 = np.where(comparefinal, comparefinal, 0)*rho1*(x1**2+y1**2+z1**2)
    vztrajn1 = np.where(comparefinal, comparefinal, 0)*rho1**2*(x1**2+y1**2+z1**2)**2
    vztrajn2 = np.sum(vztrajn1)
    J = np.sum(vztrajn0)/n
    var_J = (vztrajn2/n) - J**2
    sigma_J = V * var_J / np.sqrt(m)
    
    return V, masa, J, sigma_V, sigma_masa, sigma_J 

def analiticna(R):
        return 8*R**3*(2-np.sqrt(2))
##prvic = volumen_valji(200000000,R)*8*R**3
R = 1
napaka1 =volumen_valji2(1000000,1)[3]
drugicV = volumen_valji2(1000000,1)[0]
drugicM  = volumen_valji2(1000000,1)[1]
drugicJ = volumen_valji2(1000000,1)[2]
print(r'napake: %.5f, %.5f, %.6f'%(volumen_valji2(1000000,1)[3],volumen_valji2(1000000,1)[4],volumen_valji2(1000000,1)[5]))
'''gostota(r)'''
ro_rV = volumen_valji3(1000000,1,5)[0]
ro_rM  = volumen_valji3(1000000,1,5)[1]
ro_rJ = volumen_valji3(1000000,1,5)[2]
print(r'napake: %.5f, %.5f, %.6f'%(volumen_valji3(1000000,1,5)[3],volumen_valji3(1000000,1,5)[4],volumen_valji3(1000000,1,5)[5]))
ro_rV1 = volumen_valji3(1000000,1,20)[0]
ro_rM1  = volumen_valji3(1000000,1,20)[1]
ro_rJ1 = volumen_valji3(1000000,1,20)[2]
print(r'napake: %.5f, %.5f, %.6f'%(volumen_valji3(1000000,1,20)[3],volumen_valji3(1000000,1,20)[4],volumen_valji3(1000000,1,20)[5]))

def napaka(n,R):
    V,napaka = volumen_valji2(n,R)[0], volumen_valji2(n,R)[3]
    var = float(analiticna(R) - V)
#    print(V,napaka)
    return var, np.sqrt(np.abs(napaka))
    
stevila = np.linspace(10,1000,1000)
def fun(x,a,b):
     return a*x + b 
 

def graf(i,j,stevila,R):   
    fig=plt.figure(i)
    sub= fig.add_subplot(j)
    tocke = np.array([])
    napake= np.array([])
    drugicV = volumen_valji2(1000000,1)[0]
    for k in range(len(stevila)):
        var,prava_napaka=napaka(int(stevila[k]),R)
        
        tocke = np.append(tocke,np.log10(abs(var)))
        napake =np.append(napake, prava_napaka/drugicV)
        
    x_i = np.log10(stevila)
    a,=sub.plot(x_i,tocke,'ko')
    a,=sub.plot(x_i,tocke,'k',color='darkred', alpha=0.3)
    p0 = (-1,0)
    popt, pcov = curve_fit(fun, x_i, tocke, p0 ,napake)
    plt.text(1.5,-3,'y = b + a log(N) \n b  = %f , a = %f'%(popt[1],popt[0]))
    a,=plt.plot(x_i, fun(x_i, *popt), '--',color='darkred',label='fit')
    plt.grid()
    plt.xlabel(r'$\log_{10}{N}$')
    plt.ylabel(r'$\log_{10}{|V-V_{analitična}|}$') 
    plt.title('Razlika med simulacijo in analitično vrednostjo')
    plt.legend()
    plt.errorbar(x_i,tocke,napake,fmt='o')
    
    return fig,sub,tocke,x_i,napake,pcov
fig,sub,tocke,x_i,napake,pcov=graf(1,111,stevila,1)

#x_i = graf(1,111,stevila,1)[3]
#######################################################################################################################
'''odvisnost m(p)'''
pji = np.linspace(1,50,150)
rezultati = np.array([])
for i in range(len(pji)):
    rezultati = np.append(rezultati, volumen_valji3(1000000,1,pji[i])[2])
    print(rezultati)

plt.figure(4)
plt.grid()
plt.plot(pji,np.log(rezultati),'ko', color='darkred')
plt.plot(pji,np.log(rezultati),'ko', color='darkred',alpha=0.3)
plt.xlabel('$p$')
plt.ylabel(r'log(J [$kgm^2$])')
plt.title(r'Odvisnost vztrajnostnega momenta od parametra p za gostoto $\rho(r) = (r/R)^p$') 
  
 #######################################################################################################################
'''curve_fit'''

plt.figure(2)
plt.grid()
plt.xlabel(r'$\log_{10}{N}$')
plt.ylabel(r'$\log_{10}{|V-V_{analytic}|}$') 

b,=plt.plot(x_i,tocke,color='blue',label='$rezultati$ $metode$')
plt.legend()
plt.title('$Razlike$ $med$ $analitično$ $vrednostjo$ $in$ $rezultatom$ $metode$')
