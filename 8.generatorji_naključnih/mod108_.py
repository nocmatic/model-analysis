# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:20:43 2017

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 00:07:25 2017

@author: jure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy import optimize
import sys  
from scipy.optimize import curve_fit
from numpy.linalg import solve
from scipy.optimize import minimize
from cycler import cycler


import scipy.stats as stats

plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))




#def chi(random,dist):
#    if dist == 'enakomerno':        
#        chi= sum((random- np.ones(len(random)))**2)        
#        return chi
    
def graf(r,i,j,xlab,ylab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    
    vrednosti, bins, patches = plt.hist(r, alpha=0.3 ,edgecolor = 'black')
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab) 
    
    chi_test,p= stats.chisquare(vrednosti,  np.ones(len(vrednosti))/(len(bins)-1) * sum(vrednosti))
    #D,p_k= stats.ks_2samp(vrednosti,  np.ones(len(vrednosti)))
    sub.text(0.2,max(vrednosti)/float(2),r'$\chi^2$ = %f'%(chi_test))
    sub.text(0.2,max(vrednosti)/float(2)-max(vrednosti)/float(10),r'$p_{chi}$ = %f'%p)
    
#    sub.text(0.6,0.6,r'$D_{kolmogorov}$ = %f'%(D))
#    sub.text(0.6,0.5,r'$p_{kolmogorov}$ = %f'%p_k)
    return fig,sub,chi_test

def graf2(r1,i,j,xlab,ylab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    vrednosti, bins, patches = plt.hist(r1,bins=20, alpha=0.3 ,edgecolor = 'black',normed=True,cumulative=True)
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab) 
    x = np.ones(20)
    
    #chi_test,p= stats.chisquare(vrednosti,  np.ones(len(vrednosti)))
    D,p_k = stats.kstest(r1, 'uniform')
    
#    sub.text(0.2,0.6,r'$\chi^2$ = %f'%(chi_test))
#    sub.text(0.2,0.5,r'$p_{chi}$ = %f'%p)
    x = np.linspace(0,1,100)
    plt.plot(x,x,'r-',label='F(x)=x')
   
    sub.text(0.6,0.6,r'$D_{kolmogorov}$ = %f'%(D))

    sub.text(0.6,0.5,r'dvostranski $p_{kolmogorov}$ = %f'%p_k)

    return fig,sub,D

'''GLEDAMO HI - KVADRAT, IN TESTITRAMO ENAKOMERNO PORAZFDELITEV'''
generator= np.random.RandomState()
n = 100

r1 = generator.rand(n)
fig,sub,chi3 = graf(r1,1,121,r'razred $N_k$',r'$\frac{dN}{dN_k}$', 'Porazdelitev naključnih števil po 10 razredih')
sub.set_title('n = %e'%(n))


fig,sub,chi = graf2(r1,3,121,'naključna števila','F(x)', 'Kumulativna porazdelitev naključnih števil')
sub.set_title('n = %e'%(n))
n = 1000
r2 = generator.rand(n)
fig,sub,chi1= graf(r2,1,122,r'razred $N_k$',r'$\frac{dN}{dN_k}$')
sub.set_title('n = %e'%(n))
fig,sub,chi = graf2(r2,3,122,'naključna števila','F(x)', 'Kumulativna porazdelitev naključnih števil')

sub.set_title('n = %e'%(n))
n = 10000
r3 = generator.rand(n)
fig,sub ,chi2= graf(r3,2,121,r'razred $N_k$',r'$\frac{dN}{dN_k}$')

sub.set_title('n = %e'%(n))

fig,sub,chi = graf2(r3,4,121,'naključna števila','F(x)', 'Kumulativna porazdelitev naključnih števil')
sub.set_title('n = %e'%(n))
###############################################################################


n = 1000000
r4 = generator.rand(n)
fig,sub,chi = graf(r4,2,122,'naključna števila','število izžrebanj','Porazdelitev naključnih števil po 10 razredih')


sub.set_title('n = %e'%(n))

fig,sub,chi = graf2(r4,4,122,'naključna števila','F(x)', 'Kumulativna porazdelitev naključnih števil')
sub.set_title('n = %e'%(n))
###########################################################################################################################




'''porazdelitev statistike hi-kvadrat'''
#
def porazdelitev_chi(M, n, generator):
    '''M je število poiskusov, n pa število generiranih števil'''
    vrednosti_chi= np.array([])   

    
    for i in range( M) :
        r1 = generator.rand(n)
        vrednosti, bins = np.histogram(  r1,   bins =10)
        
        chi_test,p= stats.chisquare(vrednosti,  np.ones(len(vrednosti))/(len(bins)-1) *n)
        vrednosti_chi = np.append(vrednosti_chi,chi_test)
    return vrednosti_chi
        
generator= np.random.RandomState(seed=10)  

chiji = porazdelitev_chi(1000,100,generator)

fig=plt.figure(5)
plt.suptitle('porazdelitev $\chi^2$, prostostne stopnje = 10 - 1, število poiskusov M = 2000'   )
sub = fig.add_subplot(121)
plt.title(r'N = 100')
plt.hist(chiji,bins=40,edgecolor='black',normed=True)
x = np.linspace(0,40,100)
prava_dist= stats.chi2.pdf(x,9 )

plt.plot(x,prava_dist,'r-', label= r'teoretska krivulja')
plt.ylabel(r'$\frac{dP}{d\chi^2} $')
plt.xlabel(r'$\chi^2$')
plt.legend()

sub = fig.add_subplot(122)
chiji = porazdelitev_chi(2000,1000,generator)
plt.title(r'N = 1000')
plt.hist(chiji,bins=20,edgecolor='black',normed=True)
x = np.linspace(0,40,100)
prava_dist= stats.chi2.pdf(x,9 )

plt.plot(x,prava_dist,'r-',label=r'teoretska krivulja')
plt.ylabel(r'$\frac{dP}{d\chi^2} $')
plt.xlabel(r'$\chi^2$')
plt.legend()
########################################################################################
fig=plt.figure(6)
sub = fig.add_subplot(121)
chiji = porazdelitev_chi(2000,5000,generator)
plt.suptitle('porazdelitev $\chi^2$, prostostne stopnje = 10 - 1, število poiskusov M = 2000'   )
plt.title(r'N = 10000')
plt.hist(chiji,bins=40,edgecolor='black',normed=True)

x = np.linspace(0,20,100)
prava_dist= stats.chi2.pdf(x,9 )
plt.plot(x,prava_dist,'r-', label=r'teoretska krivulja')
plt.ylabel(r'$\frac{dP}{d\chi^2} $')
plt.xlabel(r'$\chi^2$')
plt.legend()

###############################################################################################
'''porazdelitev statistike kolmogrovih D * sqrt(N)'''
def porazdelitev_D(M, n, generator):
    '''M je število poizkusov, n pa število generiranih števil'''
    vrednosti_D= np.array([])   
    x = np.linspace(0,1,1000)
    
    for i in range( M):
        r1 = generator.rand(n)
        
        
        
        D,p_k = stats.kstest(r1, 'uniform')
        vrednosti_D = np.append(vrednosti_D,D)
    return vrednosti_D



fig=plt.figure(7)
sub = fig.add_subplot(121)
Dji = porazdelitev_D(5000,1000,generator)
plt.suptitle('porazdelitev $D\sqrt{N}$, prostostne stopnje = 10 - 1, število poiskusov M = 2000'   )
plt.title(r'N = 1000')
plt.hist(Dji,bins=40,edgecolor='black',normed=True)

x = np.linspace(0,1,100)
#prava_dist= stats.chi2.pdf(x,9 )
#plt.plot(x,prava_dist,'r-', label=r'teoretska krivulja')
plt.ylabel(r'$\frac{dP}{dD\sqrt{N}} $')
plt.xlabel(r'$D\sqrt{N}$')
plt.legend()
################################################################################
fig=plt.figure(7)
sub = fig.add_subplot(122)
Dji = porazdelitev_D(5000,10000,generator)
plt.suptitle('porazdelitev $D\sqrt{N}$, prostostne stopnje = 10 - 1, število poiskusov M = 2000'   )
plt.title(r'N = 10000')
plt.hist(Dji,bins=40,edgecolor='black',normed=True)

x = np.linspace(0,1,100)
#prava_dist= stats.chi2.pdf(x,9 )
#plt.plot(x,prava_dist,'r-', label=r'teoretska krivulja')
plt.ylabel(r'$\frac{dP}{dD\sqrt{N}} $')
plt.xlabel(r'$D\sqrt{N}$')
plt.legend()