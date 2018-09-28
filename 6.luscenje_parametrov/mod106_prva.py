# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:08:57 2017

@author: jure
"""


from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.integrate as integ
from decimal import Decimal
import math as math
from matplotlib.patches import Rectangle
from scipy import integrate
from scipy import optimize
import sys  
from scipy.optimize import linprog
from numpy.linalg import solve
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from cycler import cycler
from scipy.integrate import odeint
from scipy import optimize
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (15,5))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'g', 'y','c', 'm', 'k'])))

podatki=np.loadtxt('farmakoloski.dat',skiprows=1)
#
def graf(x,y,yerr,i,j,xlab,ylab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'ko',label='meritve',markersize=2.8)  
    a,=sub.plot(x,y,'k')  
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    
    
    plt.errorbar(x,y,yerr,fmt='ko')
      
   
    plt.legend()
    return fig,sub
def graf3(x,y,yerr,i,j,xlab,ylab,c,label,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    a,=sub.plot(x,y,'%s'%c,label= '%s'%(label))  
   
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    
    
    #plt.errorbar(x,y,yerr,fmt='ko')
      
   
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
def fit(x,a,b):
    return a + b*x
def pravi_fit(x,alfa,y0):
    return y0*x/(x+alfa)
yerr=3
x = podatki[:,0]
x_s=x
y = podatki[:,1]
y_s = y
fig, sub = graf(x,y,0,1,121,'doza reagenta X','odziv tkiva Y ')

sub.set_title('Graf odziva tkiva Y na dozo reagenta X')

'''brez napak'''
#x1 = 1/x
#y1= 1/y
#
#fig, sub = graf(x1,y1,0,1,122,'1/x','1/y')
#fig.tight_layout()
#
#
#A = np.vstack([np.ones(len(x1)),x1]).T
#res= np.linalg.lstsq(A, y1)
#a,b = res[0]
#x = np.linspace(0,1,100)
#fig,sub=graf_fit(x,fit(x,a,b),1,122)
#sub.set_title('Linearni sistem y = Ax + b')
#sub.grid()
#
#y0 = 1/a
#alfa= y0*b
#
#x = np.linspace(0,1000,100)
#fig,sub=graf_fit(x,pravi_fit(x,alfa,y0),1,121,'fitanje brez upoštevanja napak')
#fig.subplots_adjust(top=0.84)
#sub.grid()
#plt.text(400,20,'$\chi^2 = %f $ \n $\widetilde{\chi}^2 = %f $ \n $ y_0$= %f \n $ a$ = %f '%(res[1],res[1] / (len(x1)-2),y0,alfa) )
#


##########################################################
#
'''z napakami'''
yerr=3
yerr1= y1**2 *yerr
y2 = y1/yerr1
x2= x1/yerr1
y2 = y2
x2=x2
yerr1 = yerr1
A = np.vstack([np.ones(len(x2))/yerr1,x2]).T

res1= np.linalg.lstsq(A, y2)
a,b = res1[0]
x = np.linspace(0,1,100)

fig, sub = graf(x1,np.log(y1),np.log(yerr1),2,122,'1/x','log(1/y)')

graf_fit(x,np.log(fit(x,a,b)),2,122)
plt.grid()
y0 = 1/a
alfa= y0*b
fig, sub = graf(x_s,y_s,3,2,121,'doza reagenta X','odziv tkiva Y ')
x = np.linspace(0,1000,10000)
y=pravi_fit(x,alfa,y0)
fig,sub=graf_fit(x,pravi_fit(x,alfa,y0),2,121,'linearna regresija ob upoštevanju napak')
A = np.vstack([np.ones(len(x2))/yerr1,x2]).T
M = A.T.dot(A)
U,s,V=np.linalg.svd(M)
pinv_svd = np.dot(np.dot(V.T,np.linalg.inv(np.diag(s))),U.T)
da,db = np.sqrt(pinv_svd)[0][0],np.sqrt(pinv_svd)[1][1]

dy0= -y0**2*da
dalfa = (db - alfa * da )/a
sub.text(400,40,'$\chi^2 = %f $ \n $\widetilde{\chi}^2 = %f $ \ny0 = %f $\pm$ %f \n a = %f  $\pm$ %f  '%(res1[1],res1[1] / (len(x1)-2),y0,dy0,alfa,dalfa))
sub.grid()
#########################################################
#absorbcijski=np.loadtxt('CdL3_linfit.norm',skiprows=1)
#plt.figure(3)
#
#t = absorbcijski[:,0]
#
#fig,sub=graf3(t,absorbcijski[:,1],0,1,132,'energija','absorbcija na energijo','k.','Vzorec 1. Krovna plast $C.Thlaspi$')
#
#fig,sub=graf3(t,absorbcijski[:,2],0,1,133,'energija','absorbcija na energijo','k.','Vzorec 2. Sredica listov $C.Thlaspi$')
#fig,sub=graf3(t,absorbcijski[:,3],0,1,131,'energija','absorbcija na energijo','g.','Cd-O : vezava kadmija na kisik')
#fig,sub=graf3(t,absorbcijski[:,4],0,1,131,'energija','absorbcija na energijo','m.','Cd-S: vezava žvepla na sulfat', 'Absorbcijski spekter Kadmija v rastlinah in različni kemijski okolici')
#sub.grid()
#
#'''fit 1'''
#
#y = absorbcijski[:,1]
#x1 = absorbcijski[:,3]
#x2 = absorbcijski[:,4]
#
#A = np.vstack([x1,x2]).T
#res= np.linalg.lstsq(A, y)
#res1= res
#a1, a2 = res[0]
#y2 = a1* x1 + a2*x2
#
#ocena_1= res[1]/(len(y)-2) 
#
#M0 = A.T.dot(A)
#U,s,V=np.linalg.svd(M0)
#pinv_svd = np.dot(np.dot(V.T,np.linalg.inv(np.diag(s))),U.T)
#da1,da2 = np.sqrt((pinv_svd)[0][0]*ocena_1),np.sqrt((pinv_svd)[1][1]*ocena_1)#(X^T X)^-1
##da1_0,da2_0 = np.sqrt(pinv_svd)[0][0],np.sqrt(pinv_svd)[1][1]
#
#fig,sub=graf3(t,y2,0,1,132,'energija','absorbcija na energijo','r-',
#              '\n $\chi^2$ = %e \n fit: $y_{c1}= a_1 Cd_O + a_2 Cd_S$ \n $a_1 = %f \pm %f$, \n $a_2= %f \pm %f$'%(res[1],a1,da1,a2,da2))
#
#
#sub.grid()
#y= absorbcijski[:,2]
#res= np.linalg.lstsq(A, y
#                    )
#res2= res
#a1, a2 = res[0]
#y2 = a1* x1 + a2*x2
#ocena_2 = res[1]/(len(y)-2)
#da1,da2 = np.sqrt((pinv_svd)[0][0]*ocena_2),np.sqrt((pinv_svd)[1][1]*ocena_2)
#fig,sub=graf3(t,y2,0,1,133,'energija','absorbcija na energijo','r-',
#              '\n $\chi^2$ = %e \n fit: $y_{c1}= a_1 Cd_O + a_2 Cd_S$ \n $a_1 = %f \pm %f$, \n $a_2= %f \pm %f$'%(res[1],a1,da1,a2,da2))
#sub.grid()
#fig.tight_layout()
#fig.subplots_adjust(top=0.90)

############################################################################################################
#
#'''napaka 0.02'''
#absorbcijski=np.loadtxt('CdL3_linfit.norm',skiprows=1)
#plt.figure(3)
#
#t = absorbcijski[:,0]
#
#fig,sub=graf3(t,absorbcijski[:,1],0,1,132,'energija','absorbcija na energijo','k.','Vzorec 1. Krovna plast $C.Thlaspi$')
#
#fig,sub=graf3(t,absorbcijski[:,2],0,1,133,'energija','absorbcija na energijo','k.','Vzorec 2. Sredica listov $C.Thlaspi$')
#fig,sub=graf3(t,absorbcijski[:,3],0,1,131,'energija','absorbcija na energijo','g.','Cd-O : vezava kadmija na kisik')
#fig,sub=graf3(t,absorbcijski[:,4],0,1,131,'energija','absorbcija na energijo','m.','Cd-S: vezava žvepla na sulfat', 'Absorbcijski spekter Kadmija v rastlinah in različni kemijski okolici')
#sub.grid()
#
#'''fit 1'''
#yerr = 0.013
#y = absorbcijski[:,1]/yerr
#x1 = absorbcijski[:,3]/yerr
#x2 = absorbcijski[:,4]/yerr
#
#A = np.vstack([x1,x2]).T
#res= np.linalg.lstsq(A, y)
#res1= res
#a1, a2 = res[0]
#y2 = a1* x1 *yerr+ a2*x2*yerr
#
#ocena_1= res[1]/(len(y)-2) 
#
#M0 = A.T.dot(A)
#U,s,V=np.linalg.svd(M0)
#pinv_svd = np.dot(np.dot(V.T,np.linalg.inv(np.diag(s))),U.T)
#da1,da2 = np.sqrt((pinv_svd)[0][0]*ocena_1),np.sqrt((pinv_svd)[1][1]*ocena_1)#(X^T X)^-1
##da1_0,da2_0 = np.sqrt(pinv_svd)[0][0],np.sqrt(pinv_svd)[1][1]
#
#fig,sub=graf3(t,y2,0,1,132,'energija','absorbcija na energijo','r-',
#              '\n $\chi^2$ = %e \n fit: $y_{c1}= a_1 Cd_O + a_2 Cd_S$ \n $a_1 = %f \pm %f$, \n $a_2= %f \pm %f$'%(res[1],a1,da1,a2,da2))
#
#
#sub.grid()
#yerr = 0.018
#x1 = absorbcijski[:,3]/yerr
#x2 = absorbcijski[:,4]/yerr
#
#y= absorbcijski[:,2]/yerr
#A = np.vstack([x1,x2]).T
#res= np.linalg.lstsq(A, y
#                    )
#res2= res
#a1, a2 = res[0]
#y2 = a1* x1*yerr + a2*x2*yerr
#ocena_2 = res[1]/(len(y)-2)
#da1,da2 = np.sqrt((pinv_svd)[0][0]*ocena_2),np.sqrt((pinv_svd)[1][1]*ocena_2)
#fig,sub=graf3(t,y2,0,1,133,'energija','absorbcija na energijo','r-',
#              '\n $\chi^2$ = %e \n fit: $y_{c1}= a_1 Cd_O + a_2 Cd_S$ \n $a_1 = %f \pm %f$, \n $a_2= %f \pm %f$'%(res[1],a1,da1,a2,da2))
#sub.grid()
#fig.tight_layout()
#fig.subplots_adjust(top=0.90)
#
#    
