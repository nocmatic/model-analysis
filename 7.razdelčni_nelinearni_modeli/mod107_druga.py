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



#def narediAiksi(izmerki,napake,funkcije):
#    matrika = []
#    for i,j in zip(izmerki,napake):
#        vrstica = [f(i)/j for f in funkcije]
#        matrika.append(vrstica)
#    return np.matrix(matrika)
#def ita(dekompozicija,b,i):
#    return np.sum(dekompozicija[0][:,i]*b)/dekompozicija[1][i] * dekompozicija[2][i]
 



def graf(x,y,yerr,i,j,xlab,ylab,plot_lab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'ko',label='%s'%plot_lab,markersize=2.8)  
    a,=sub.plot(x,y,'k',alpha=0.5)  
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    
    
    plt.errorbar(x,y,yerr,fmt='ko')
      
   
    plt.legend()
    return fig,sub
def graf_fit(x,y,i,j,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'r--',alpha=0.5,label='fit')  
     
  
    plt.legend()
    return fig,sub

############################################################################################
'''le en kompartment'''
def ledvice_1(t,c0,lamb):
    return  c0*np.exp(-lamb*t)
def chi(fun,x,y,yerr,popt):
    a1,a2 = popt    
    chi= sum((fun(x,a1,a2)-y)**2/yerr**2)        
    return chi
def chi_plot(fun,x,y,yerr,popt):
    a1,a2 = popt    
    chi= (fun(x,a1,a2)-y)**2/yerr**2      
    return chi
izmerki = np.loadtxt('ledvice.dat',skiprows=1)
x = izmerki[:,0]
y = izmerki[:,1]
p0 = (10324.4466034, 0)
yerr = np.array( len(x) *[50])
popt,pcov = curve_fit(ledvice_1,x,y,p0,sigma=yerr)
c0,lamb = popt
print(c0,lamb)

chi1 = chi(ledvice_1,x,y,yerr,popt)
chi1_r= chi1/(len(x)-2)
x2 = np.linspace(-100,100,100)
var = np.sqrt(np.diagonal(pcov))


graf(x,y,yerr,6,121,'Čas[s]','N - število sunkov','meritve')
fig,sub =graf_fit(x,ledvice_1(x,c0,lamb),6,121)
sub.grid()
sub.set_title(r'le en kompartment $ y= A e^{-\lambda_1 t}$, brez ozadja')
plt.text(1000,10000, ' $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A$ = %.5f $\pm$ %.5f \n $\lambda_1$ = %.5f $\pm$ %.5f \n '%(yerr[0],chi1,chi1_r,popt[0],var[0],popt[1],var[1]))


fig,sub = graf(x,chi_plot(ledvice_1,x,y,yerr,popt),yerr,9,111,'izmerek','Napaka fita $\chi_i^2$','napake')
sub.grid()
##########################################################################
'''dodatek aditivne konstante'''
'''le en kompartment'''
def ledvice_1(t,c0,lamb,a):
    return  c0*np.exp(-lamb*t) +a 
def chi(fun,x,y,yerr,popt):
    a1,a2,a3 = popt    
    chi= sum((fun(x,a1,a2,a3)-y)**2/yerr**2)        
    return chi
x = izmerki[:,0]
y = izmerki[:,1]
p0 = (1000,0.001,1000)
popt1,pcov1 = curve_fit(ledvice_1,x,y,p0,sigma=yerr)
c0,lamb,a = popt1
popt5= popt1
print(c0,lamb,a)
popt= popt1

chi2 = chi(ledvice_1,x,y,yerr,popt1)
chi2_r= chi2/(len(x)-3)

var1 = np.sqrt(np.diagonal(pcov1))
var =var1

graf(x,y,yerr,6,122,'Čas[s]','N - število sunkov','meritve')
fig,sub =graf_fit(x,ledvice_1(x,c0,lamb,a),6,122)
sub.grid()
sub.set_title(r'Kompartment + ozadje $ y= A e^{-\lambda_1 t} + B$, ')
plt.text(1000,10000, ' $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A$ = %.5f $\pm$ %.5f \n $\lambda_1$ = %.5f $\pm$ %.5f \n $B$= %.5f $\pm$ %.5f '%(yerr[0],chi2,chi2_r,popt[0],var[0],popt[1],var[1],popt[2],var[2]))

#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
def chi_plot(fun,x,y,yerr,popt):
    a1,a2,a3 = popt    
    chi= (fun(x,a1,a2,a3)-y)**2/yerr**2      
    return chi
fig,sub = graf(x,chi_plot(ledvice_1,x,y,yerr,popt),[0]*len(y),10,111,'izmerek','Napaka fita $\chi_i^2$','napake')
sub.grid()

########################################################################################################
def ledvice_1(t,c0,lamb,a,c1,lamb1):
    return  c0*np.exp(-lamb*t) +a  + c1 * np.exp(-lamb1*t)

def half1(t,c0,lamb,a):
    return  c0*np.exp(-lamb*t) +a
def half2(t,a,c1,lamb1):
    return  a  + c1 * np.exp(-lamb1*t)

def chi(fun,x,y,yerr,popt):
    a1,a2,a3,a4,a5 = popt    
    chi= sum((fun(x,a1,a2,a3,a4,a5)-y)**2/yerr**2)        
    return chi
x = izmerki[:,0]
y = izmerki[:,1]
p0 = (1000,0.001,1000,1000,0.001)
popt1,pcov1 = curve_fit(ledvice_1,x,y,p0,sigma=yerr)
c0,lamb,a,c1,lamb1 = popt1
print(c0,lamb,a)
popt= popt1

chi2 = chi(ledvice_1,x,y,yerr,popt1)
chi2_r= chi2/(len(x)-4)

var1 = np.sqrt(np.diagonal(pcov1))
var =var1

graf(x,y,yerr,7,121,'Čas[s]','N - število sunkov','meritve')
fig,sub =graf_fit(x,ledvice_1(x,c0,lamb,a,c1,lamb1),7,121)
sub.grid()
sub.set_title(r'Dva kompartmenta $ y= A_1 e^{-\lambda_1 t}+ A_2 e^{-\lambda_2 t}+ B$, ')
plt.text(1000,5000, ' $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A_1$ = %.5f $\pm$ %.5f \n $\lambda_1$ = %.5f $\pm$ %.5f \n $B$= %.5f $\pm$ %.5f \n $A_2$  %.5f $\pm$ %.5f \n $\lambda_2$ =  %.5f $\pm$ %.5f  '%(yerr[0],chi2,chi2_r,popt[0],var[0],popt[1],var[1],popt[2],var[2],popt[3],var[3],popt[4],var[4]))



x = np.linspace(0,2300,100)
sub.plot(x,half1(x,c0,lamb,a),'b',label='$y= A_1 e^{-\lambda_1 t} + b$')
sub.plot(x,half2(x,a,c1,lamb1),'g',label='$y =  A_2 e^{-\lambda_2 t}+ B$')
sub.legend()
sub.legend()
#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
x= izmerki[:,0]
fig.tight_layout()
def chi_plot(fun,x,y,yerr,popt):
    a1,a2,a3,a4,a5 = popt    
    chi= (fun(x,a1,a2,a3,a4,a5)-y)**2/yerr**2      
    return chi
fig,sub = graf(x,chi_plot(ledvice_1,x,y,yerr,popt),[0]*len(y),11,111,'izmerek','Napaka fita $\chi_i^2$','napake')
sub.grid()


###########################################################


#
#def ledvice_2(t,c0,lamb,a,c1,lamb1):
#    return  c0*np.exp(-lamb*np.sqrt(t)) +a  + c1 * np.exp(-lamb1*np.sqrt(t))
#
#def half2(t,c0,lamb,a):
#    return  c0*np.exp(-lamb*np.sqrt(t)) +a
#def half3(t,a,c1,lamb1):
#    return  a  + c1 * np.exp(-lamb1*np.sqrt(t))
#
#def chi(fun,x,y,yerr,popt):
#    a1,a2,a3,a4,a5 = popt    
#    chi= sum((fun(x,a1,a2,a3,a4,a5)-y)**2/yerr**2)        
#    return chi
#x = izmerki[:,0]
#y = izmerki[:,1]
#p0 = popt
#popt3,pcov1 = curve_fit(ledvice_2,x,y,p0,sigma=yerr)
#c0,lamb,a,c1,lamb1 = popt3
#print(c0,lamb,a)
#popt= popt1
#
#chi2 = chi(ledvice_2 ,x,y,yerr,popt1)
#chi2_r= chi2/(len(x)-3)
#
#var1 = np.sqrt(np.diagonal(pcov1))
#var =var1
#
#graf(x,y,yerr,7,122,'Čas[s]','N - število sunkov','meritve')
#fig,sub =graf_fit(x,ledvice_2(x,c0,lamb,a,c1,lamb1),7,122)
#sub.grid()
#sub.set_title(r'Dva kompartmenta $ y= A_1 e^{-\lambda_1  sqrt t}+ A_2 e^{-\lambda_2 sqrt t }+ B$, ')
#plt.text(1000,5000, ' $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A_1$ = %.5f $\pm$ %.5f \n $\lambda_1$ = %.5f $\pm$ %.5f \n $B$= %.5f $\pm$ %.5f \n $A_2$  %.5f $\pm$ %.5f \n $\lambda_2$ =  %.5f $\pm$ %.5f  '%(yerr[0],chi2,chi2_r,popt[0],var[0],popt[1],var[1],popt[2],var[2],popt[3],var[3],popt[4],var[4]))
#
#
#
#x = np.linspace(0,2300,100)
#sub.plot(x,half2(x,c0,lamb,a),'b',label='$y= A_1 e^{-\lambda_1 t} + b$',alpha=0.3)
#sub.plot(x,half3(x,a,c1,lamb1),'g',label='$y =  A_2 e^{-\lambda_2 t}+ B$',alpha=0.3)
#sub.legend()
#sub.legend()
##sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
##sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')
#
#fig.tight_layout()
####################################################################

def ledvice_3(t,c0,lamb,a):
    return  c0*np.exp(-lamb*np.sqrt(t)) +a 

def half2(t,c0,lamb,a):
    return  c0*np.exp(-lamb*np.sqrt(t)) +a
def half3(t,a,c1,lamb1):
    return  a  + c1 * np.exp(-lamb1*np.sqrt(t))

def chi(fun,x,y,yerr,popt):
    a1,a2,a3 = popt    
    chi= sum((fun(x,a1,a2,a3)-y)**2/yerr**2)        
    return chi
x = izmerki[:,0]
y = izmerki[:,1]
p0 = popt5
popt1,pcov1 = curve_fit(ledvice_3,x,y,p0,sigma=yerr)
c0,lamb,a= popt1
print(c0,lamb,a)
popt= popt1

chi2 = chi(ledvice_3 ,x,y,yerr,popt1)
chi2_r= chi2/(len(x)-3)

var1 = np.sqrt(np.diagonal(pcov1))
var =var1

graf(x,y,yerr,7,122,'Čas[s]','N - število sunkov','meritve')
fig,sub =graf_fit(x,ledvice_3(x,c0,lamb,a),7,122)
sub.grid()
sub.set_title(r'Korenski model $ y= A e^{-\lambda_1  sqrt(t)}+ B$, ')
plt.text(1000,5000, ' $y_{err}=%.5f$ \n $chi^2 $= %.5f \n $chi_{reduced}^2 $= %.5f \n $A_1$ = %.5f $\pm$ %.5f \n $\lambda_1$ = %.5f $\pm$ %.5f \n $B$= %.5f $\pm$ %.5f  '%(yerr[0],chi2,chi2_r,popt[0],var[0],popt[1],var[1],popt[2],var[2]))



x = np.linspace(0,2300,100)
#sub.plot(x,half2(x,c0,lamb,a),'b',label='$y= A_1 e^{-\lambda_1 t} + b$',alpha=0.3)
#sub.plot(x,half3(x,a,c1,lamb1),'g',label='$y =  A_2 e^{-\lambda_2 t}+ B$',alpha=0.3)
sub.legend()
sub.legend()
#sub.plot(x2,half_1(x2,I0,Ua,Uc),'b',label='$I_0e^{U/Ua}$')
#sub.plot(x2,half_2(x2,I0,Ua,Uc),'g',label='$-I_0e^{-U/Uc}$')

fig.tight_layout()
