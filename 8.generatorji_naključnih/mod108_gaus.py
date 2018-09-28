# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:38:02 2017

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

import scipy.special as spec
import scipy.stats as stats
import timeit
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'darkred', 'darkred', 'y','c', 'm', 'k'])))




def gausskum(x,size=1):
    return 0.5*(1+spec.erf(x/np.sqrt(2)))

def graf(r,i,j,xlab,ylab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    
    vrednosti, bins, patches = plt.hist(r, bins = 20, alpha=0.3 ,edgecolor = 'black',normed=True)
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab) 
    D,p_k = stats.kstest(r, 'norm')
    sub.text(0,0.2,r'$D_{kolmogorov}$ = %f'%(D))

    sub.text(0,0.1,r'dvostranski $p$ = %f'%p_k)
    x = np.linspace(bins[0],bins[-1],1000)
    y = stats.norm.pdf(x) 
    plt.plot(x,y,'r--',label='Gausovka')
    plt.legend()
    return fig,sub


'''ta naredi guasova števila'''
from numpy import log,sin,cos,sqrt,pi
def box_muller(u1,u2):
  z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
  z2 = sqrt(-2*log(u1))*sin(2*pi*u2)
  return z1,z2
#######################################################################################################
generator= np.random.RandomState()
n = 1000
r1 = generator.rand(n)

n1 = generator.rand(n)
n2 = generator.rand(n)
z1,z2 = box_muller(n1,n2)
fig,sub = graf(z1,1,121,r'razred $N_k$',r'$\frac{dN}{dN_k}$', 'Gausovo porazdeljena naključna števila (10 razredov) N = 1000 ')
sub.set_title('Box Muller')

########################################################
def konvolucijska(size=1):
    if size==1:
        return np.sum(np.random.rand(6)) - np.sum(np.random.rand(6))
    else:
        return [konvolucijska() for i in range(size)]
 

gauss_konvolucijski=[]
for i in range(500): 
    gauss_konvolucijski.append(konvolucijska())
fig,sub = graf(gauss_konvolucijski,1,122,r'razred $N_k$',r'$\frac{dN}{dN_k}$', )
sub.set_title('Konvolucjiski generator')
a,b = stats.kstest(konvolucijska,gausskum,N=1000)


########################################################################
##############################################################################
n = 10000 #nov n!!!
r1 = generator.rand(n)

n1 = generator.rand(n)
n2 = generator.rand(n)
z1,z2 = box_muller(n1,n2)
fig,sub = graf(z1,2,121,r'razred $N_k$',r'$\frac{dN}{dN_k}$', 'Gausovo porazdeljena naključna števila (10 razredov) N = 10000 ')
sub.set_title('Box Muller')


def konvolucijska(size=1):
    if size==1:
        return np.sum(np.random.rand(6)) - np.sum(np.random.rand(6))
    else:
        return [konvolucijska() for i in range(size)]
 


gauss_konvolucijski=np.array([])
for i in range(10000): 
   gauss_konvolucijski=np.append( gauss_konvolucijski,konvolucijska())
fig,sub = graf(gauss_konvolucijski,2,122,r'razred $N_k$',r'$\frac{dN}{dN_k}$', )
sub.set_title('Konvolucjiski generator')
a,b = stats.kstest(konvolucijska,gausskum,N=10000)
################################################################################################
# compute binary search time
def cas_twiser(n):
    SETUP_CODE = '''
from __main__ import box_muller
import scipy.stats as stats
import numpy as np'''
     
    TEST_CODE = '''
generator= np.random.RandomState()
n = 10000
r1 = generator.rand(n)
    
n1 = generator.rand(n)
n2 = generator.rand(n)
z1,z2 = box_muller(n1,n2)
    '''
         
    # timeit.repeat statement
    times = timeit.repeat(setup = SETUP_CODE,
                              stmt = TEST_CODE,
                              
                              number = 1000)
 
    # priniting minimum exec. time
    print('Čas za %f iteracij: %f'%(n,min(times))) 
    return min(times)

#casovna_zahtevnost_twiser=[]
#for i in range(0,10000,1000):
#    casovna_zahtevnost_twiser.append(cas_twiser(i))
#plt.figure(3)
#x = np.linspace(0,10000,10000/1000)
#plt.plot(x,casovna_zahtevnost_twiser,'ro',markersize=2.5,label='Box Muller Twiser generator')


##################################################################################################
'''plot theree numbers'''
def get_box_muller(n):
    
    
    n1 = generator.rand(n)
    n2 = generator.rand(n)
    z1,z2 = box_muller(n1,n2)
    
    
def get_convolutional(n):
    gauss_konvolucijski=np.array([])
    for i in range(n): 
       gauss_konvolucijski=np.append( gauss_konvolucijski,konvolucijska())
    return gauss_konvolucijski
if __name__=='__main__':
    from timeit import Timer
    casi = []
    for n in range(0,100000,100):
        t = Timer(lambda: get_box_muller(n))
        a=t.repeat(number=1,repeat=1)
       
        c = min(a)
        casi.append(c)
        print (c)
        
if __name__=='__main__':
    from timeit import Timer
    casi2 = []
    for n in range(0,10000,100):
        t = Timer(lambda: get_convolutional(n))
        a=t.repeat(number=1,repeat=1,)
       
        c = min(a)
        casi2.append(c)
        print (c)
##  
plt.figure(4)      
x = np.linspace(0,100000,100000/100)
plt.plot(x,casi,'r',label='Box Muller Twiser generator')
plt.xlabel('Število naključnih števil - n')
plt.ylabel(r'cas $( s)$')
x = np.linspace(0,10000,10000/100)
plt.plot(x,casi2,'b',label='Konvolucijski generator')
plt.legend()
plt.grid()