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
from scipy.stats import poisson

from scipy.optimize import curve_fit



import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 12))

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

#%%
###########################################################################################################
#
#       1. NALOGA - ENOSTAVNO EKSPONENTNO
#
###########################################################################################################

#def umirajoča(n,beta,N):
#    rešitev=[]
#    for i in range(n):
#        if N>0:
#            rešitev.append(N)
#            N=N-np.random.poisson(N*beta)
#        else:
#            rešitev.append(0)
#    return [i for i in range(n)],rešitev
#
#def rojevajoča(n,beta,N):
#    rešitev=[]
#    for i in range(n):
#        if N>0:
#            rešitev.append(N)
#            N=N+np.random.poisson(N*beta)
#        else:
#            rešitev.append(0)
#    return [i for i in range(n)],rešitev
#
#def rojstvosmrt(n,betaR,betaS,t,N):
#    rešitev=[]
#    for i in range(n):
#        if N>0:
#            rešitev.append(N)
#            N=N+np.random.poisson(N*betaR*t)-np.random.poisson(N*t*betaS)
#        else:
#            rešitev.append(0)
#    return [i for i in range(n)],rešitev
#
#
##%%
##RIŠEM VEČ PRIMEROV UMIRAJOČIH
#plt.figure(1)
#for i in range(6):
#    reši=umirajoča(100,0.05*i+0.01,100)
#    plt.plot(reši[0],reši[1],label=r'$\beta \Delta t={0:.2f}$'.format(0.05*i+0.01))
#plt.title(r'$N_0 = 25$')
#plt.xlabel(r'Število korakov $\Delta t$')
#plt.ylabel('N')
#plt.legend(loc=0)
#plt.grid()
##
##%%
##RIŠEM VEČ ISTIH UMIRAJOČIH
#fig = plt.figure(2)
#sub1 = fig.add_subplot(131)
#resi_mean = np.zeros(300)
#for i in range(5):
#    reši=umirajoča(300,0.02,25)
#    plt.plot(reši[0],reši[1])
#    resi_mean += reši[1]
#resi_mean = resi_mean / float(5)
#plt.plot(reši[0],resi_mean,'r--')
#
#plt.title(r' $N_0 = 25$, $\beta \Delta t = 0.02$')
#plt.xlabel(r'Število korakov $\Delta t$')
#plt.ylabel('N')
#plt.legend(loc=0)
#plt.grid()
#sub2 = fig.add_subplot(132)
#
#
#
#resi_mean = np.zeros(300)
#for i in range(5):
#    reši=umirajoča(300,0.02,100)
#    plt.plot(reši[0],reši[1])
#    resi_mean += reši[1]
#resi_mean = resi_mean / float(5)
#plt.plot(reši[0],resi_mean,'r--')
#plt.title(r' $N_0 = 100$, $\beta \Delta t = 0.02$')
#plt.xlabel(r'Število korakov $\Delta t$')
#plt.ylabel('N')
#plt.legend(loc=0)
#plt.grid()
#sub2 = fig.add_subplot(133)
#resi_mean = np.zeros(300)
#for i in range(5):
#    reši=umirajoča(300,0.02,250)
#    plt.plot(reši[0],reši[1])
#    resi_mean += reši[1]
#resi_mean = resi_mean / float(5)
#plt.plot(reši[0],resi_mean,'r--')
#plt.title(r' $N_0 = 250$, $\beta \Delta t = 0.02$')
#plt.xlabel(r'Število korakov $\Delta t$')
#plt.ylabel('N')
#plt.legend(loc=0)
#plt.grid()
#fig.tight_layout()
#
##plt.title(r'Povprečje in fit eksponetne rešitve $N_0 = 25$, $\beta \Delta t = 0.02$')
##plt.plot(reši[0],resi_mean,'b--',label='povprečje')
##
##def func(t,a,b): 
##    return a * np.exp(-1*b * t)
##popt,pcov = curve_fit(func,reši[0], resi_mean,p0=(0,0))
##plt.plot(reši[0],func(np.array(reši[0]),*popt),'r--',label='fit')
##plt.legend()
##plt.grid()
##fig.tight_layout()
##plt.text(100,60,r'$N = N_0 e^{- \beta t}$'+'\n'+r' $N_0 = %.4f \pm %.5f$'%(popt[0],np.sqrt(pcov[0][0])) +'\n' + r'$\beta = %.4f \pm %.5f$'%( popt[1], np.sqrt(pcov[1][1])))
##plt.xlabel(r'Število korakov $\Delta t$')
##%%
##RIŠEM VEČ PRIMEROV ROJEVAJOČIH
plt.figure(3)
for i in range(5):
    reši=rojevajoča(100,0.05*i+0.01,5)
    plt.plot(reši[0],np.log(reši[1]),label=r'$\beta \Delta t ={0:.2f}$'.format(0.05*i+0.01))
plt.title(r'Rojevanje $N_0 = 1i$')
plt.xlabel(r'Število korakov $\Delta t$')
plt.ylabel('log(N)')
plt.legend(loc=0)
plt.grid()
#
###%%
##RIŠEM VEČ PRIMEROV ROJEVAJOČIH IN UMIRAJOČIH
#for i in range(6):
#    reši=rojstvosmrt(600,4,5,0.05*i+0.01,250)
#    plt.plot(reši[0],reši[1],label='razmerje beta*dt t={0:.2f}'.format(0.05*i+0.01))
#plt.title('Začetna populacija = 250. Model rojevajočih in umirajočih')
#plt.xlabel('Število časovnih korakov')
#plt.ylabel('Število osebkov v populaciji N')
#plt.legend(loc=0)
##
###%%
###RIŠEM VEČ ISTIH ROJEVAJOČIH IN UMIRAJOČIH PRI ISTIH BETA SO RAZLICNI ZARADI STATISTICNE
#fig= plt.figure(6)
#
#sub2 = fig.add_subplot(121)
#for i in range(10):
#    reši=rojstvosmrt(600,5,5,0.01,100)
#    plt.plot(reši[0],reši[1])
#plt.title(r'Rojstvo + smrt, $N_0 = 100$  $\beta_r = \beta_s$')
#plt.xlabel(r'Število korakov $\Delta t$')
#plt.ylabel('N ')
#plt.legend(loc=0)
#plt.grid()
#fig.tight_layout()
##
###%%
#############################################################################################################
###
###       1. NALOGA - II. del STATISTIKA ČASOV IZUMRTJA - ENOJNA
###
#############################################################################################################
#
#def stat_umirajoča(n,beta,t,N):
#    rešitev=[]
#    for j in range(n):
#        i=0
#        število=N
#        while število > 0:
#            število=število-np.random.poisson(N*beta*t)
#            i+=1
#        rešitev.append(i*t)
#    return [j for j in range(n)],rešitev
#
##%%
#HISTOGRAM IZOMTRJA. SPREMENI ŠTEVILO TEH GAUSOVK ;*
plt.figure(7)
for i in range(7):
    rešitev=stat_umirajoča(5000,0.05,250)
    plt.hist(rešitev[1],bins=100,alpha=0.75,label=r'$\beta \Delta t={0:.2f},$'.format(0.1*i+0.1)+r'$ \Delta t = 0.1$' )
plt.title('Statistika izumtrja N = 250, ')
plt.xlabel(r'$t_{izumrtje}$')
plt.ylabel('Število dogodkov')
plt.legend(loc=0)
plt.grid()
#%%
#POVPREČNI ČAS IZUMRTJA V ODVISNOSTI OD BETA - UMIRAJOČI MODEL 
plt.figure(8)
povprečje=[]
N=40
    
povprečje=[]
bete=[]
for i in range(N):
        rešitev=stat_umirajoča(5000,0.1,0.05*i+0.1,250)
        povprečje.append(np.mean(rešitev[1]))
        bete.append(0.05*i+0.1)
plt.title(r'Povprečni čas izumrtja v odvisnosti od konstante $\beta$, $\Delta t = 0.1$')
plt.plot(bete,povprečje,'r+',label=r'')
    #plt.plot([i for i in range(N)],povprečje,'')
plt.xlabel('vrednost beta')
plt.ylabel('Povprečni čas izumrtja')
plt.legend(loc=0)
plt.grid()

#%%
#POVPREČNI ČAS IZUMRTJA V ODVISNOSTI OD dt - UMIRAJOČI MODEL
plt.figure(9)
povprečje=[]
N=200
for i in range(3):
    povprečje=[]
    for k in range(N):
        rešitev=stat_umirajoča(5000,0.01*i+0.01,0.01*k+0.001,100)
        povprečje.append(np.mean(rešitev[1]))
    plt.title(r'Povprečni čas izumrtja N = 100 v odvisnosti od $\Delta t$')
    plt.plot([0.001*i+0.001 for i in range(N)],povprečje,'+',label=r'$\beta$={0:.2f}'.format(0.01*i+0.01))
    plt.plot([0.001*i+0.001 for i in range(N)],povprečje,':')
plt.xlabel(r' $\Delta t$')
plt.ylabel('Povprečni čas izumrtja')
plt.legend(loc=0)
plt.grid()
#%%
############################################################################################################
##
##       1. NALOGA - STATISTIKA ČASOV IZUMRTJA - DVOJNA
##
############################################################################################################
#
#def stat_umirajoča(n,beta,t,N):
#    rešitev=[]
#    for j in range(n):
#        i=0
#        število=N
#        while število > 0:
#            število=število-np.random.poisson(N*beta*t/2)-np.random.poisson(N*beta*t/2)
#            i+=1
#        rešitev.append(i*t)
#    return [j for j in range(n)],rešitev
#
##%%
##RIŠEM HISTOGRAM ZA ČASE IZUMRTJA - UMIRAJOČI MODEL - DVOJNA
#plt.figure(10)
#for i in range(5):
#    rešitev=stat_umirajoča(5000,(0.05*i+0.1)*3,0.1,100)
#    plt.hist(rešitev[1],bins=100,alpha=0.75,label='beta={0:.2f}'.format((0.1*i+0.1)*3))
#plt.title('Statistika izumtrja N = 100, model umirajoči ')
#plt.xlabel(r'$t_{izumrtje}$')
#plt.ylabel('Število dogodkov')
#plt.legend(loc=0)
#
##%%
############################################################################################################
##
##       1. NALOGA - STATISTIKA ČASOV IZUMRTJA - ROJSTVO SMRT
##
############################################################################################################
#
#def stat_rojstvosmrt(n,betaR,betaS,t,N):
#    rešitev=[]
#    for j in range(n):
#        i=0
#        število=N
#        
#        while število > 0:
#            število=število-np.random.poisson(N*betaS*t)+np.random.poisson(N*betaR*t)
#            i+=1
#    
#        rešitev.append(i*t)
#    return [j for j in range(n)],rešitev
#
##%%
##RIŠEM HISTOGRAM ZA ČASE IZUMRTJA - UMIRAJOČI MODEL
#
#for i in range(4):
#    t=0.05*i+0.01
#    rešitev=stat_rojstvosmrt(5000,8,10,t,100)
#    plt.hist(rešitev[1],bins=100,alpha=0.5,label=r'$\Delta t = %.3f$'%(t))
#plt.title(r'Umiranje, rojstvo $\beta_r = 8, \beta_s = 10, N = 100$')
#plt.xlabel(r'$t_{izumrtje}$')
#plt.ylabel('Dogodki')
#plt.legend(loc=0)
#plt.grid()
#
#
##%%
##POVPREČNI ČAS IZUMRTJA V ODVISNOSTI OD dt - UMIRAJOČI MODEL
#povprečje=[]
#N=200
#for i in range(N):
#    rešitev=stat_rojstvosmrt(5000,4,5,0.01*i+0.01,25)
#    povprečje.append(np.mean(rešitev[1]))
#plt.title('Povprečni čas izumrtja N = 25. Model umirajočih')
#plt.plot([0.01*i+0.01 for i in range(N)],povprečje,'.')
#plt.plot([0.01*i+0.01 for i in range(N)],povprečje,':')
#plt.xlabel('vrednost dt')
#plt.ylabel('Povprečni čas izumrtja')
#plt.legend(loc=0)


#%%
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
##
##       2. NALOGA - MATRIKA PREHODOV ZA ZGORNJI MODEL
##
############################################################################################################
############################################################################################################
##      konstantni imirljivosti in smrtnosti
############################################################################################################
plt.ylabel(r'$(\frac{dP}{dN})_i$')
#%%
###########################################################################################################
#      linearno odvisni imirljivosti in smrtnosti
###########################################################################################################

def Matrika_prehodovL(N,lamb,mu,dt):
    if (lamb+mu)*N*dt<1:
        return [[1-(lamb+mu)*(j)*dt if i==j else lamb*dt*(j) if i-1==j else mu*dt*(j) if i+1==j else 0 for j in range(N)] for i in range(N)]
    else:
        return False
    
def vektorL(N,š):
    return [0 if i!=š-1 else 1 for i in range(N)]

def vektor_nL(N,lamb,mu,dt,n,š):
    b=np.array(vektorL(N,š))
    A=np.array(Matrika_prehodovL(N,lamb,mu,dt))
    for i in range(n):
        b=A.dot(b)
        vsota=sum(b)
        b=[b[i]/vsota for i in range(N)]
    return b
        
 
#zgled
#1-(lamb+mu)*n*dt        mu*dt*n             0              0
#   lamb*dt*n       1-(lamb+mu)*dt*n      mu*dt*n           0
#      0               lamb*dt*n      1-(lamb+mu)*dt*n    mu*dt*n
    
# mu so smrti
# lamb so rojstva
 
#%%
    

import seaborn as sns
sns.set_palette(sns.color_palette("winter", 20))

plt.figure(2)
N=100
stanje=[i for i in range(N)]
for i in range(20):
        plt.plot(stanje,vektor_nL(N,0,0.01,1,500*i**2+100,100),'--')
        #plt.plot(stanje,vektor_nL(N,0,5,0.01,150*i**2+100,100),'--')
plt.plot([100,100],[0,0.15],'b:')
plt.title('$\Delta t$=0.01')
plt.xlabel(r'$N_i$')
plt.ylabel(r'$(\frac{dP}{dN})_i$')
#%%
###########################################################################################################
#      povprecna vrednost raznih porazdelitev
###########################################################################################################

import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 10))

def vektor_nR(N,lamb,mu,dt,n,š):
    b=np.array(vektorL(N,š))
    A=np.array(Matrika_prehodovL(N,lamb,mu,dt))
    seznam=[]
    števec=[]
    for i in range(n):
        b=A.dot(b)
        vsota=sum(b)
        b=[b[i]/vsota for i in range(N)]
        if i%100:
            povpr=sum(b[i]*i for i in range(N))
            seznam.append(povpr)
            števec.append(i)
    return števec,seznam

N=150
n=10000

povprečje=[]

plt.figure(4)

stanje=[i*0.0001 for i in range(n)]
resi=vektor_nR(N,4,10,0.0001,n,100)
plt.plot(resi[0],resi[1],'r-',label='umirajoča pop.= 10/4')
resi=vektor_nR(300,5,5,0.0001,n,100)
plt.plot(resi[0],resi[1],'g-',label='nevtralna pop.= 5/5')
resi=vektor_nR(500,10,4,0.0001,n,100)
plt.plot(resi[0],resi[1],'b-',label='rojevajoča pop.= 4/10')

plt.title('Povprečne vrednosti za razne primere dt=0.0001')
plt.xlabel('čas')
plt.ylabel('Povprečno število osebkov')
plt.legend()

#%%
###########################################################################################################
#      varianca raznih porazdelitev
###########################################################################################################

import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 10))

def vektor_nV(N,lamb,mu,dt,n,š):
    b=np.array(vektorL(N,š))
    A=np.array(Matrika_prehodovL(N,lamb,mu,dt))
    seznam=[]
    števec=[]
    for i in range(n):
        b=A.dot(b)
        vsota=sum(b)
        b=[b[i]/vsota for i in range(N)]
        if i%100:
            povpr=sum(b[i]*i for i in range(N))
            var=sum((i-povpr)*(i-povpr)*b[i] for i in range(N))
            seznam.append(var)
            števec.append(i)
    return števec,seznam

N=150
n=10000

povprečje=[]

plt.figure(5)

stanje=[i*0.0001 for i in range(n)]
resi=vektor_nV(N,4,10,0.0001,n,100)
plt.plot(resi[0],resi[1],'r-',label='umirajoča pop.= 10/4')
resi=vektor_nV(N,5,5,0.0001,n,100)
plt.plot(resi[0],resi[1],'g-',label='nevtralna pop.= 5/5')
resi=vektor_nV(N,10,4,0.0001,n,100)
plt.plot(resi[0],resi[1],'b-',label='rojevajoča pop.= 4/10')

plt.title('Varianca za razne primere dt=0.0001')
plt.xlabel('čas')
plt.ylabel('Varianca')
plt.legend()


#%%
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
#
#       2. NALOGA - MATRIKA PREHODOV ZA ZGORNJI MODEL
#
###########################################################################################################
###########################################################################################################
#      začetni model 5/4
###########################################################################################################




def zajci_lisica(n,a,b,Zo,Lo,dt):
    zajci=[]
    lisice=[]
    Z=Zo
    L=Lo
    for i in range(n):
        if Z>0 and L>0:
            zajci.append(Z)
            lisice.append(L)
            Z=Z+np.random.poisson(5*a*Z*dt)-np.random.poisson(4*a*Z*dt)-np.random.poisson(a/Lo*Z*L*dt)
            L=L+np.random.poisson(4*b*L*dt)-np.random.poisson(5*b*L*dt)+np.random.poisson(b/Zo*Z*L*dt)
        else:
            zajci.append(Z)
            lisice.append(L)
    return [i*dt for i in range(n)],zajci,lisice


#%%
#RIŠEM FAZNI DIAGRAM
    
import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 2))
plt.figure(10)

for i in range(1):
    reši=zajci_lisica(50000,1,2,200,50,0.01)
    
    plt.figure(0)
    plt.plot(reši[1],reši[2],'r')#label='razmerje beta*dt t={0:.2f}'.format(0.05*i+0.01))
    plt.title('Fazni diagram zajci-lisice')
    plt.xlabel('Število zajcev')
    plt.ylabel('Število lisic')
    
    plt.figure(1)
    plt.plot(reši[0],reši[1],'g:',label='Zajci')
    plt.plot(reši[0],reši[2],'r:',label='Lisice')
    plt.title('Odvisnost populacij od časa')
    plt.xlabel('Čas')
    plt.ylabel('Število osebkov')
    plt.legend(loc=0)

#%%
#ALGORITEM ZA RISANJE LEPIH
plt.figure(11)
def zajci_lisicaHUD(n,a,b,Zo,Lo,dt):
    zajci=[]
    lisice=[]
    Z=Zo
    L=Lo
    for i in range(n):
        if Z>0 and L>0:
            zajci.append(Z)
            lisice.append(L)
            Zp=Z
            Z=Z+np.random.poisson(5*a*Z*dt)-np.random.poisson(4*a*Z*dt)-np.random.poisson(a/Lo*Z*L*dt)
            L=L+np.random.poisson(4*b*L*dt)-np.random.poisson(5*b*L*dt)+np.random.poisson(b/Zo*Zp*L*dt)
            j=i
        else:
            zajci.append(Z)
            lisice.append(L)
    return [i*dt for i in range(n)],zajci,lisice,j*dt
    
    
import seaborn as sns
sns.set_palette(sns.color_palette("autumn", 2))

reši=zajci_lisicaHUD(10000,1,2,200,50,0.01)
while reši[3]<60:
    reši=zajci_lisicaHUD(10000,1,2,200,50,0.01)

#%%
plt.figure(12)
plt.plot(reši[1],reši[2],'r')#label='razmerje beta*dt t={0:.2f}'.format(0.05*i+0.01))
plt.title('Fazni diagram zajci-lisice')
plt.xlabel('Število zajcev')
plt.ylabel('Število lisic')
    
plt.figure(13)
plt.plot(reši[0],reši[1],'g:',label='Zajci')
plt.plot(reši[0],reši[2],'r:',label='Lisice')
plt.title('Odvisnost populacij od časa')
plt.xlabel('Čas')
plt.ylabel('Število osebkov')
plt.legend(loc=0)


zajc=np.abs(np.fft.fft([reši[1][i] for i in range(5000)]))
lisc=np.abs(np.fft.fft([reši[2][i] for i in range(5000)]))
plt.figure(2)
plt.title('Fourijejeva transformacija za preimer zajcev')
plt.plot([i for i in range(len(zajc))],zajc)
plt.xlabel('Valovno število')
plt.ylabel('Amplituda')
plt.figure(3)
plt.title('Fourijejeva transformacija za preimer lisic')
plt.plot([i for i in range(len(lisc))],lisc)
plt.xlabel('Valovno število')
plt.ylabel('Amplituda')

#%%

vrhovi=[2.1,5.4,9.3,14.29,18,20.9,24.64,29.3,32.9,37,39.5,43.6,48.7,52.6,57.2]

razmaki=[(vrhovi[i+1]-vrhovi[i]) for i in range(len(vrhovi)-1)]

povprecen_razmak=sum(razmaki)/len(razmaki)
print(povprecen_razmak)

#%%
###########################################################################################################
#
#      3. NALOGA povprečna življenska doba običajne populacije 
#
###########################################################################################################
plt.figure(15)
def zajci_lisicaHUD2(n,a,b,Zo,Lo,dt):
    Z=Zo
    L=Lo
    i=0
    while i<n and Z>0 and L>0:
        Zp=Z
        Z=Z+np.random.poisson(5*a*Z*dt)-np.random.poisson(4*a*Z*dt)-np.random.poisson(a/Lo*Z*L*dt)
        L=L+np.random.poisson(4*b*L*dt)-np.random.poisson(5*b*L*dt)+np.random.poisson(b/Zo*Zp*L*dt)
        j=i
        i=i+1
    return j*dt

#%%
N=10**4
sistema=[]
for i in range(N):
    d=zajci_lisicaHUD2(10000,1,2,200,50,0.01)
    sistema.append(d)
#%%


plt.title('Statistika življenskih dob')
plt.hist(sistema,150,alpha=0.75)
plt.xlabel('Življenska doba')
plt.ylabel('Število dogodkov')


print(sum(sistema)/len(sistema))



#
#
#
#
#
#
#
#
#
#
#
#
#








