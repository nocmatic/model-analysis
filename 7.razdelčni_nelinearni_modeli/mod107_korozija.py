# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:28:44 2017

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


def fun(x, I0,Ua,Uc):
     return I0*(np.exp(x/Ua) - np.exp(-x/Uc))
#def grad(x,I0,Ua,Uc):
def half_1(x,I0,Ua,Uc):
    return I0*(np.exp(x/Ua))
def half_2(x,I0,Ua,Uc):
    return -I0*np.exp(-x/Uc)
def chi(fun,x,y,yerr,popt):
    I0,Ua,Uc= popt
    
    chi= sum((fun(x,I0,Ua,Uc)-y)**2/yerr**2)
        
    return chi
def graf(x,y,yerr,i,j,xlab,ylab,plot_lab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'ko',label='%s'%plot_lab,markersize=2.8)  
    a,=sub.plot(x,y,'k--',alpha=0.5)  
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    
    plt.legend()
    #plt.errorbar(x,y,yerr,fmt='o')
      
   

    return fig,sub
def graf2(x,y,yerr,i,j,xlab,ylab,plot_lab,co,c,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'%s'%co,label='%s'%plot_lab,markersize=2.8)  
    a,=sub.plot(x,y,'%s--'%c,alpha=0.5)  
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    
    plt.legend()
    #plt.errorbar(x,y,yerr,fmt='o')
      
   

    return fig,sub
def graf_fit(x,y,i,j,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'--',alpha=0.5,label='fit')  
     
  
    plt.legend()
    return fig,sub
############################################################################################
podatki=np.loadtxt('korozija.txt')
x= podatki[:,0]
x_org = x       #to so original podatki za ksneje
xpovp= x.mean() #povprečje x
sigmax= x.std() #standardni odklon
x = (x-xpovp)/sigmax #normalizacija za kasneje
y = podatki[:,1]

####################################################################################
'''fit je močno odvisen od začetnega približka'''

'''uporabimo originalne podatke torej x_org'''

p0=[1,2,2]
yerr =np.array(len(y)*[0.0002])
popt3, pcov3 = curve_fit(fun, x_org, y,  p0 ,sigma = yerr,method='lm')
I0,Ua,Uc= popt3

chi3 = chi(fun,x_org,y,yerr,popt3)
chi3_r= chi3/(len(x)-3)
x2 = np.linspace(-100,100,100)
var = np.sqrt(np.diagonal(pcov3))


graf(x_org,y,yerr,6,111,'U[V]','I[A]','meritve')
fig,sub =graf_fit(x_org,fun(x_org,I0,Ua,Uc),6,111)
sub.grid()
#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')










    
############################################################################################################
'''fit z normalizacijo hitrejše konvergira  https://en.wikipedia.org/wiki/Feature_scaling -- vse bo na figure(1)'''
p0 = [1,1,1]
yerr =np.array(len(y)*[0.0002])
popt, pcov = curve_fit(fun, x, y,  p0 ,sigma = yerr,method='lm')

I0,Ua,Uc= popt
fig = plt.figure(1)
sub = fig.add_subplot(121)
plt.xlabel(r'$\widetilde{U}$ ')
plt.ylabel('I[A]')
x1= np.linspace(min(x)-2,max(x)+2,100)
plt.plot(x,y,'k',alpha=0.5)
plt.plot(x,y,'ko',label='meritve')
#plt.plot(x-xpovp,y,'r',label='x-<x>')
plt.plot(x1,fun(x1,I0,Ua,Uc),'r--',label='fit',alpha=0.5)
plt.plot(x1,half_1(x1,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
plt.plot(x1,half_2(x1,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
sub.set_title(r'fit z normalizacijo $U =\frac{ U - <U> }{\sigma}$')
plt.grid()
chi1 = chi(fun,x,y,yerr,popt)
chi1_r= chi1/(len(x)-3)

var = np.sqrt(np.diagonal(pcov))
plt.text(0,-0.03, '$I= I_0(e^{U/U_a}- e^{-U/U_c})$  \n zacetni priblizek =%s \n $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n $U_c$= %.5f $\pm$ %.5f \n <U> = %.5f \n $\sigma$ = %.5f'%(p0,yerr[0],chi1,chi1_r,popt[0],var[0],popt[1],var[1],popt[2],var[2],xpovp,sigmax))
sub.legend()


var = np.sqrt(np.diagonal(pcov))
def korelacije(pcov, var):
    korelacije=[pcov[0][1],pcov[1][2],pcov[0][2]]
    
    korelacijski = [ korelacije[0] /(var[0]*var[1]) , korelacije[1] / (var[2]*var[1]), korelacije[2] / (var[0]*var[2]) ]
    return korelacijski
a = korelacije(pcov,var)


#####################################################################################################
'''reskaliranje podatkov med [0,1] --- ne deluje dobro!!! GLEJ  https://en.wikipedia.org/wiki/Feature_scaling'''
#def rescalling(x):
#    xmax = max(x)
#    xmin= min(x)
#    return (x-xmin)/(xmax-xmin)
#x_norm= rescalling(x_org)
#p0 = [1,1,1]
#yerr =np.array(len(y)*[0.0002])
#popt1, pcov1 = curve_fit(fun, x_norm, y,  p0 ,sigma = yerr,method='lm')
#chi2 = chi(fun,x_norm,y,yerr,popt1)
#chi2_r= chi2/(len(x)-3)
#
#I0,Ua,Uc= popt1
#graf(x_norm,y,yerr,1,122,r'$\widetilde{U}$ ','I[A]','meritve')
#fig,sub =graf_fit(x_norm,fun(x_norm,I0,Ua,Uc),1,122)
#sub.set_title(r'normalizacija $U = \frac{U - <U>}{max(U) - min(U)}$')
#sub.text(0.6,-0.002, ' \n  $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f'%(chi2,chi2_r))
#fig.tight_layout()
#sub.legend()
#sub.grid()
#
#########################################################################################################
#'''uvedem še tretjo normalizacijo'''
#def normalization3(x):
#    xmax = max(x)
#    xmin= min(x)
#    xmean = x.mean()
#    return (x-xmean)/(xmax-xmin)
#x_norm2= normalization3(x_org)
#p0 = [1,1,1]
#yerr =np.array(len(y)*[0.0002])
#popt6, pcov6 = curve_fit(fun, x_norm2, y,  p0 ,sigma = yerr,method='lm')
#chi2 = chi(fun,x_norm2,y,yerr,popt6)
#chi2_r= chi2/(len(x)-3)
#
#I0,Ua,Uc= popt6
#graf(x_norm2,y,yerr,8,121,r'$\widetilde{U}$ ','I[A]','meritve')
#fig,sub =graf_fit(x_norm2,fun(x_norm2,I0,Ua,Uc),8,121)
#sub.set_title(r'normalizacija $U = \frac{U - <U>}{max(U) - min(U)}$')
#sub.text(0,-0.004, ' \n  $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f'%(chi2,chi2_r))
#fig.tight_layout()
#sub.legend()
#sub.grid()
#################################################################################################################
'''Sedaj uporabim začetni približek od normalizacije, pomnožen s sigmo_x :  popt * sigma GRAF 2'''
x = x_org

p0= popt*sigmax     #začetni približek
yerr =np.array(len(y)*[0.0002])
popt3, pcov3 = curve_fit(fun, x, y,  p0 ,sigma = yerr,method='lm')
I0,Ua,Uc= popt3
graf(x,y,yerr,3,121,r'$\widetilde{U}$','I[A]','meritve')
chi3 = chi(fun,x,y,yerr,popt3)
chi3_r= chi3/(len(x)-3)
x2 = np.linspace(-150,150,100)
var = np.sqrt(np.diagonal(pcov3))

fig,sub =graf_fit(x2,fun(x2,I0,Ua,Uc),3,121)
sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')


sub.text(0.5,-0.015, '$I= I_0(e^{U/U_a}- e^{-U/U_c})$ \n p0 = %.3f,%.3f,%.3f \n   $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n  $U_c$ = %.5f $\pm$ %.5f'%(p0[0],p0[1],p0[2],chi3,chi3_r,popt3[0],var[0],popt3[1],var[1],popt3[2],var[2]))
sub.set_title('Osnovni model')
sub.legend()
sub.grid()
def chi_plot(fun,x,y,yerr,popt):
    I0,Ua,Uc= popt
    
    chi= (fun1(x,I0,Ua,Uc,U0)-y)**2/yerr**2
    return chi
fig,sub = graf2(x,chi_plot(fun,x,y,yerr,popt3),[0]*len(y),13,111,'izmerek','Napaka fita $\chi_i^2$','brez $U_0$','ro','r')

################################################################################################
'''dodamo U0 za boljši fit? GRAF 2'''

def fun1(x, I0,Ua,Uc,U0):
     return I0*(np.exp((x-U0)/Ua) - np.exp(-(x-U0)/Uc))
#def grad(x,I0,Ua,Uc):
def half_11(x,I0,Ua,Uc,U0):
    return I0*(np.exp((x-U0)/Ua))
def half_21(x,I0,Ua,Uc,U0):
    return -I0*np.exp(-(x-U0)/Uc)
def chi1(fun,x,y,yerr,popt):
    I0,Ua,Uc,U0= popt
    
    chi= sum((fun1(x,I0,Ua,Uc,U0)-y)**2/yerr**2)
    return chi
'''začetni približek: od popt * sigma'''
x = x_org
p0=np.append(popt*sigmax,xpovp)

yerr =np.array(len(y)*[0.0001])
popt4, pcov4 = curve_fit(fun1, x, y,  p0 ,sigma = yerr,method='lm')
I0,Ua,Uc,U0= popt4
graf(x,y,yerr,3,122,'U[V]','I[A]','meritve')
chi4 = chi1(fun,x,y,yerr,popt4)
chi4_r= chi4/(len(x)-4)
x2 = np.linspace(-150,150,100)

var = np.sqrt(np.diagonal(pcov4))
fig,sub =graf_fit(x2,fun1(x2,I0,Ua,Uc,U0),3,122)
sub.plot(x2,half_11(x2,I0,Ua,Uc,U0),'b',label='$I_0e^{U/Ua}$')
sub.plot(x2,half_21(x2,I0,Ua,Uc,U0),'g',label='$-I_0e^{-U/Uc}$')


sub.text(0.5,-0.015, '$I= I_0(e^{(U-U_0)/U_a}- e^{-(U-U_0)/U_c})$ \n p0=%.3f,%.3f,%.3f,%.3f  \n  $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n  $U_c$ = %.5f $\pm$ %.5f \n $U_0$ = %.5f $\pm$ %.5f '%(p0[0],p0[1],p0[2],p0[3],chi4,chi4_r,popt4[0],var[0],popt4[1],var[1],popt4[2],var[2],popt4[3],var[3]))

sub.set_title('Dodaten parameter $U_0$')
sub.legend()
sub.grid()
def chi_plot(fun,x,y,yerr,popt):
    I0,Ua,Uc,U0= popt
    
    chi= (fun1(x,I0,Ua,Uc,U0)-y)**2/yerr**2
    return chi
fig,sub = graf2(x,chi_plot(fun1,x,y,yerr,popt4),[0]*len(y),13,111,'izmerek','Napaka fita $\chi_i^2$','napake z $U_0$','bo','b')
sub.grid()
#############################################################################################################################
#'''test konvergence z drugo normalizacijo (Feature : popt1) --- ni dober približek p0'''
#x = x_org
#
#p0= popt1*(max(x)-min(x))   #začetni približek
#yerr =np.array(len(y)*[0.0002])
#popt3, pcov3 = curve_fit(fun, x, y,  p0 ,sigma = yerr,method='lm')
#I0,Ua,Uc= popt3
#graf(x,y,yerr,7,122,'x','y','meritve')
#chi3 = chi(fun,x,y,yerr,popt3)
#chi3_r= chi3/(len(x)-3)
#x2 = np.linspace(-150,150,100)
#var = np.sqrt(np.diagonal(pcov3))
#
#fig,sub =graf_fit(x2,fun(x2,I0,Ua,Uc),7,122)
#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
#
#
#sub.text(0.5,-0.015, '$I= I_0(e^{U/U_a}- e^{-U/U_c})$ \n p0 = %.3f,%.3f,%.3f \n   $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n  $U_c$ = %.5f $\pm$ %.5f'%(p0[0],p0[1],p0[2],chi3,chi3_r,popt3[0],var[0],popt3[1],var[1],popt3[2],var[2]))
#sub.set_title('Reskaliranje: ni dober približek')
#
#sub.grid()
########################################################################################################################
#
#'''test konvergence s TRETJO normalizacijo '''
#x = x_org
#
#p0= popt6*(max(x)-min(x))   #začetni približek
#yerr =np.array(len(y)*[0.0002])
#popt3, pcov3 = curve_fit(fun, x, y,  p0 ,sigma = yerr,method='lm')
#I0,Ua,Uc= popt3
#graf(x,y,yerr,7,121,'x','y','meritve')
#chi3 = chi(fun,x,y,yerr,popt3)
#chi3_r= chi3/(len(x)-3)
#x2 = np.linspace(-150,150,100)
#var = np.sqrt(np.diagonal(pcov3))
#
#fig,sub =graf_fit(x2,fun(x2,I0,Ua,Uc),7,121)
#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
#
#
#sub.text(0.5,-0.015, '$I= I_0(e^{U/U_a}- e^{-U/U_c})$ \n p0 = %.3f,%.3f,%.3f \n   $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n  $U_c$ = %.5f $\pm$ %.5f'%(p0[0],p0[1],p0[2],chi3,chi3_r,popt3[0],var[0],popt3[1],var[1],popt3[2],var[2]))
#sub.set_title('Normalizacija 3. slab začetni približek')
#
#sub.grid()

###############################################################################################
def polinom(x,A,B,C):
    return A*x + B*x**2 + C*x**3
    
p0=[1,2,2]
yerr =np.array(len(y)*[0.0002])
popt3, pcov3 = curve_fit(polinom, x_org, y,  p0 ,sigma = yerr,method='lm')
A,B,C= popt3

chi3 = chi(polinom,x_org,y,yerr,popt3)
chi3_r= chi3/(len(x)-3)
x2 = np.linspace(-100,100,100)
var = np.sqrt(np.diagonal(pcov3))

A = popt3[0]
B = popt3[1]
C = popt3[2]
Ua = (B/A - np.sqrt(6*C/A -B**2/(A**2)))**(-1)
Uc =  (-B/A - np.sqrt(6*C/A -B**2/(A**2)))**(-1)
I0 = A/(np.sqrt(24*C/A - (2*B/A)**2))

graf(x_org,y,yerr,11,121,'U[V]','I[A]','meritve')
fig,sub =graf_fit(x_org,polinom(x_org,A,B,C),11,121, r'fit iz razvoja :$ I = AU + B U^2 + C U^3$')
sub.grid()
var = np.sqrt(np.diagonal(pcov3))
plt.text(40,-0.004, '$I = AU + B U^2 + C U^3$  \n zacetni priblizek =%s \n $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A$ = %.5f $\pm$ %.5f \n $B$ = %.5f $\pm$ %.5f \n $C$= %.5f $\pm$ %.5f \n <U> = %.5f$'%(p0,yerr[0],chi3,chi3_r,popt3[0],var[0],popt3[1],var[1],popt3[2],var[2],xpovp))
plt.text(-25,-0.009, r'$I_0$= %.5f,$U_a$ = %.5f, $U_c$ = %.5f'%(-I0,Ua,Uc,))
sub.legend()
sub.set_title('razvita rešitev hitro konvergira')
#############################################################################
'''test novega pribliška iz starega DELA'''
p0 = [I0,Ua,Uc]
x = x_org

 #začetni približek
yerr =np.array(len(y)*[0.0002])
popt3, pcov3 = curve_fit(fun, x, y,  p0 ,sigma = yerr,method='lm')
I0,Ua,Uc= popt3
graf(x,y,yerr, 11,122,'x','y','meritve')
chi3 = chi(fun,x,y,yerr,popt3)
chi3_r= chi3/(len(x)-3)
x2 = np.linspace(-150,150,100)
var = np.sqrt(np.diagonal(pcov3))

fig,sub =graf_fit(x2,fun(x2,I0,Ua,Uc),11,122)
sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')

sub.grid()
sub.text(0.5,-0.015, '$I= I_0(e^{U/U_a}- e^{-U/U_c})$ \n p0 = %.3f,%.3f,%.3f \n   $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $I_0$ = %.5f $\pm$ %.5f \n $U_a$ = %.5f $\pm$ %.5f \n  $U_c$ = %.5f $\pm$ %.5f'%(p0[0],p0[1],p0[2],chi3,chi3_r,popt3[0],var[0],popt3[1],var[1],popt3[2],var[2]))
sub.set_title('Približek iz razvitega modela uporabimo na pravem modelu')

