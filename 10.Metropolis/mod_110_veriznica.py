# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:50:30 2017

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
plt.rc('font', size = 17, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (18,8))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))
def start_position(n):    
    generator= np.random.RandomState()
    matrika = np.array([])   
    zacetni_stolpec = np.zeros(n)
    koncni_stolpec = np.zeros(n)
    zacetni_stolpec[0] = 1
    koncni_stolpec[0]=1
    matrika = np.append(matrika,zacetni_stolpec)
    for i in range(n-2):
        n1 = int(generator.rand()*(n))
        
        stolpci = np.zeros(n)
        stolpci[n1] = 1
      
        matrika =  np.column_stack((matrika,stolpci))      
    matrika = np.column_stack((matrika,koncni_stolpec))
    return matrika
def prava_veriznica(n,m):
    generator= np.random.RandomState()
    matrika = np.array([])   
    zacetni_stolpec = np.zeros(m)
    koncni_stolpec = np.zeros(m)
    zacetni_stolpec[0] = 1
    koncni_stolpec[0]=1
    matrika = np.append(matrika,zacetni_stolpec)
    for i in range(n-2):
        n1 = int(generator.rand()*(m))
        
        stolpci = np.zeros(m)
        stolpci[n1] = 1
      
        matrika =  np.column_stack((matrika,stolpci))      
    matrika = np.column_stack((matrika,koncni_stolpec))
    return matrika

def energy(matrika,alpha):
    n = len(matrika.T)
    E=0
    hi = 0   
    st_pr=0
    for i in range(n):
        stolpec = matrika[:,i]
        hj = np.nonzero(stolpec)
        hj = hj[0]
        '''potencialna'''
        dEp = alpha * (-1*hj)
        E = E + dEp                        
        '''sedaj še izračunamo elastično energije od drugega člena dalje'''
        if i != 0 :
            dEpr=  0.5 * (-1* hj - (-1*hi))*(-1*hj - (-1* hi))
            
            E = E+ dEpr
            hi = hj
            st_pr +=1
            '''shranimo si vrednost prejšne velikosti'''
    return E[0]


def random_change(matrika,alpha):
    '''naključni element izberemo in mu spremenimo smer, prva naključno število ižreba kateri stolpec,
    drugo pa premik gor ali dol (-1,1)'''
    c = np.array(matrika)
    generator= np.random.RandomState()
    n1 = int(generator.rand()*(len(matrika.T)-2)+1)    
    n2 = np.random.randint(2) * 2 -1  #+1 ali -1 
    
    stolpec = c[:,n1]
    
    indeks_visine= np.nonzero(stolpec)
    indeks_visine = indeks_visine[0]
    
    '''na prejsni visini sedaj ni tocke'''
    stolpec[indeks_visine] = 0
    
    if indeks_visine == len(matrika)-1:
           
           n2 = -1 
           
           stolpec[indeks_visine + n2] = 10
           
           
    if indeks_visine==0:
        
        n2 = 1
        stolpec[indeks_visine + n2] = 1
    else:
        
         stolpec[indeks_visine + n2] = 1   
         
   
    
    #############################################
    '''sedaj izračunamo še spremembo energije; potrebujemo sosednji višini'''
    stolpec_levi = c[:,n1-1]
    stolpec_desni = c[:,n1+1]
    
    indeks_visine_levi= np.nonzero(stolpec_levi)
    levi = -1*indeks_visine_levi[0]    
    indeks_visine_desni= np.nonzero(stolpec_desni)
    
    desni = -1 *indeks_visine_desni[0]
    novi =-1*( indeks_visine + n2)
    stari = -1 * indeks_visine
    
    deltaE = alpha*(novi - stari)+ 0.5 *( (desni- novi)**2 + (levi-novi)**2) - 0.5*((levi-stari)**2 + (desni - stari)**2)
   
    return c,deltaE



def metropolis(zacetno_stanje,n,T0,alpha):
    '''n - število iteracij, T0 = temperatura, alpha = 1 '''
    #T = np.linspace(T0,1,n)
    generator= np.random.RandomState() 
    koncno_stanje= np.array(zacetno_stanje)
    zacetna_energija = energy(zacetno_stanje,1)
    koncna_energija = 0
    energije = np.array([])  
    
    for i in range(n):
        print('iteracija:',i)
        
        if i ==0:
            koncna_energija = koncna_energija + zacetna_energija
        poskus,deltaE=  random_change(koncno_stanje,alpha)
        if deltaE < 0:
            koncno_stanje= poskus
            koncna_energija = koncna_energija + deltaE
            if i >= 15000:
                energije = np.append(energije,koncna_energija)
            
        
            
        
        else : 
            n1 = generator.rand()
            if T0[i] == 0:
                boltzman = 0
            else:
                boltzman = np.exp(-1*deltaE / (T0[i]))
            
            if boltzman >= n1:
                koncno_stanje = poskus
                koncna_energija = koncna_energija + deltaE
           
                if i >= 15000:
                    energije = np.append(energije,koncna_energija)
            else:
                koncno_stanje = koncno_stanje 
                #energije = np.append(energije,koncna_energija)
                
                continue
        
           
    return koncno_stanje, energije, koncna_energija
def crte(resitev):
    x=np.array([])
    y=np.array([])
    for i in range(17):
        x = np.append(x,i)        
        y = np.append(y,np.nonzero(resitev[:,i])[0])
    return x,y

###########################################################################
'''dela...navadna verizica '''
#a = prava_veriznica(17,17)
#n = 10000
#alpha =1
#T3 = np.array(np.ones(10000)*[0])
#resitev3,energije3,E3 = metropolis(a,n,T3,alpha) 
#fig = plt.figure(7)
#sub = fig.add_subplot(121)
#x,y = crte(resitev3)
#sub.plot(x, y, '--', color='lightgrey')
#sub.spy(resitev3, marker = 'o', color='black', markersize = 5)
#plt.ylim(max(y)+1,0)
#plt.text(5,max(y)/2,'Energija: %.2f'%(E3))
#plt.grid()
#
#
#plt.title(r'T = %.3f, n = %d'%(T3[0],n))
#sub.set_aspect('auto')
#sub = fig.add_subplot(122)
#sub.plot(energije3)
#plt.xlabel('Iteracija')
#plt.ylabel('Energija')
#plt.title('Odvisnost energije od iteracije')
#fig.tight_layout()
#plt.grid()
#plt.ylim(-150,100)
#fig.suptitle('Omejena verižica')
#fig.subplots_adjust(top=0.9)

##########################################################

    
    
#######################################################################

'''razsirjena visina...dela'''
a = prava_veriznica(17,100)
n = 20000
alpha =1
T3 = np.array(np.ones(n)*[0.00001])
resitev3,energije3,E3 = metropolis(a,n,T3,alpha) 
fig = plt.figure(7)
sub = fig.add_subplot(121)
x,y = crte(resitev3)
sub.plot(x, y, '--', color='lightgrey')
sub.spy(resitev3, marker = 'o', color='black', markersize = 5)
plt.ylim(max(y)+1,0)
plt.text(5,max(y)/2,'Energija: %.2f'%(E3))
plt.grid()


plt.title(r'T = %.3f, n = %d'%(T3[0],n))
sub.set_aspect('auto')
sub = fig.add_subplot(122)
sub.plot(energije3)
plt.xlabel('Iteracija')
plt.ylabel('Energija')
plt.title('Odvisnost energije od iteracije')
fig.tight_layout()
plt.grid()
#plt.ylim(E3-10,2500)
fig.suptitle('Neomejena verižica')
fig.subplots_adjust(top=0.9)

#############################################################################
'''plot verižnice, torej narisat točke iz arraya na graf in nato povezat...'''
#############################################################################
'''spremenimo alpha '''
#a = prava_veriznica(17,100)
#n = 30000
#alpha =1
#T3 = np.array(np.ones(n)*[10])
#resitev3,energije3,E3,i = metropolis(a,n,T3,alpha) 
#fig = plt.figure(7)
#sub = fig.add_subplot(121)
#
#x,y = crte(resitev3)
#sub.plot(x, y, '--', color='lightgrey')
#sub.spy(resitev3, marker = 'o', color='black', markersize = 5)
#plt.ylim(max(y)+1,0)
#plt.text(5,max(y)/2,'Energija: %.2f'%(E3))
#plt.grid()
#
#
#plt.title(r'T = %.3f, n = %d'%(T3[0],n))
#sub.set_aspect('auto')
#sub = fig.add_subplot(122)
#sub.plot(energije3)
#plt.xlabel('Iteracija')
#plt.ylabel('Energija')
#plt.title('Odvisnost energije od iteracije')
#fig.tight_layout()
#plt.grid()
#
#fig.suptitle('Neomejena verižica')
#fig.subplots_adjust(top=0.9)
#plt.grid()
#
#print()
#plt.figure(10)
#Emin= np.array([])
#alpha_array= np.array([0.1,1,2,3,4,5,10,15,20])
#
#
#for i in range(len(alpha_array)-1):
#    alpha = alpha_array[i]
#    T3 = np.array(np.ones(10000)*[0.00001])
#    resitev3,energije3,E3 = metropolis(a,n,T3,alpha) 
#    Emin = np.append(Emin,E3)
#    
#plt.plot(alpha_array,Emin,'r--')
#
#    



##############################################################################################

'''simulirano ohlajanje'''

#def temperatura(T0,Tk,st_iteracij,st_):
    
#############################################################################################
#E = np.array([])
#k = 100
#n= 30000
#T = np.linspace(10,0.0001,k)
#alpha= 1
#a = prava_veriznica(17,100)
##for i in range(k):
##     
##     resitev3,energije3,E3 = metropolis(a,n,np.ones(n)*[T[i]],alpha) 
##     E3 = energije3.mean()
##     E = np.append(E,E3)
#plt.figure(2)
#plt.plot(T,E,'r+')
#plt.xlabel('kT')
#plt.ylabel('Energija')
#
#plt.grid()
#plt.title('Odvisnost končnega stanja od temperature, pri n = 30 000 iteracijah')
#
#
#def func(x,A,B):
#    return A*x + B
#
#popt,pcov = curve_fit(func,T,E,p0=(20,-120))
#plt.plot(T,func(T,*popt),'b--',label='fit')
#plt.text(2,-120,'E = A T + B \n A = %.3f + %.3f , B  = %.3f + %.3f '%(popt[0],np.sqrt(pcov[0][0]),popt[1],np.sqrt(pcov[1][1])))
#
#












    
    
        
        
    
