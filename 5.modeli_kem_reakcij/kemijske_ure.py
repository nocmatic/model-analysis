# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:23:12 2017

@author: ASUS
"""
import matplotlib.pyplot as plt
import numpy as np
import math as math
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
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k','r','g','b'])))

'''x[0] je snov I-, x[1] je snov I2, x[2] je snov S2O3'''

def graf(res,cas,i,j,p1,p2,*args):  
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(cas,res[:,0],'r',label='$I^{-}$')
    b,=sub.plot(cas,res[:,1],'g',label='$I_2$')
    c,=sub.plot(cas,res[:,2],'b',label='$S_2O_3^{2-}$')
    sub.axhline(y=8,linestyle='-',color='darkred', label='$S_2O_8^{2-}$')
    k1=p2/p1
    
#    sub.set_title(r'$[HBr(0)]= %d$'%(res[0,2]))
#    sub.set_title(r'$\frac{\widetilde{q}}{\widetilde{p}}= %d$, $[I^{-}(0)]=%d,$ $[I_2(0)]=%d$'%(k1,res[0,0],res[0,1]))
    sub.set_title(r'$\lambda= %d$'%(k1))
    sub.legend(loc='best')
    
  
    sub.grid()    
    plt.xlabel('čas[s]')
    plt.ylabel('koncentracija')
    return fig,sub
def graf_ura(res,cas,i,j,p1,p2,label,*args):  
    fig=plt.figure(i)
    lam = p2/p1
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(cas,res[:,1],label='%s'%label)
  
#    sub.set_title(r'$[HBr(0)]= %d$'%(res[0,2]))
#    sub.set_title(r'$\frac{\widetilde{q}}{\widetilde{p}}= %d$, $[I^{-}(0)]=%d,$ $[I_2(0)]=%d$'%(k1,res[0,0],res[0,1]))
    
    sub.legend(loc='best')
    
  
    sub.grid()    
    plt.xlabel('čas[s]')
    plt.ylabel('koncentracija')
    
    return fig,sub
def model(x,cas,p1,p2):
    persulfat = 10
    return np.array([-2*p1*x[0]*persulfat + 2*p2*x[1]*x[2],
                     p1*x[0]*persulfat - p2*x[2]*x[1],
                     -2*p2*x[1]*x[2]])

    
'''Grafi'''
#x0 = np.array([1,0,2])
#cas = np.linspace(0,0.81,100)
#p1,p2 = 1,1
#res1 = odeint(model,x0,cas,args=(p1,p2))
#a=res1[:,1]
#graf(res1,cas,1,131,p1,p2,)
#
#
#cas = np.linspace(0,0.81,100)
#p1,p2 = 1,100
#res2 = odeint(model,x0,cas,args=(p1,p2))
#b=res2[:,1]
#graf(res2,cas,1,132,p1,p2,r' $[S_2O_3^{2-}(0)]=2$, $[I^{-}(0)]=1,$ $[I_2(0)]=0$,')
#
#
#cas = np.linspace(0,0.81,100)
#p1,p2 = 1,2000
#res3 = odeint(model,x0,cas,args=(p1,p2))
#c=res3[:,1]
#graf(res3,cas,1,133,p1,p2)    
#########################################################
#
#x0 = np.array([1,0,10])
#cas = np.linspace(0,5,100)
#p1,p2 = 1,2000
#res2 = odeint(model,x0,cas,args=(p1,p2))
#a=res2[:,1]
#fig,sub=graf(res2,cas,2,131,p1,p2,r'Različno razmerje joddidnih ionov in tiosulfata)
#sub.set_title('$I^{-}(0)$=%d, $\lambda$ = 2000'%x0[0])
#x0 = np.array([5,0,10])
#cas = np.linspace(0,0.81,100)
#p1,p2 = 1,2000
#res2 = odeint(model,x0,cas,args=(p1,p2))
#b=res2[:,1]
#fig,sub=graf(res2,cas,2,132,p1,p2,r'Različne koncentracije jodidnih ionov na začetku')
#sub.set_title('$I^{-}(0)$=%d, $\lambda$ = 2000'%x0[0])
#x0 = np.array([20,0,10])
#cas = np.linspace(0,0.81,100)
#p1,p2 = 1,2000
#res2 = odeint(model,x0,cas,args=(p1,p2))
#c=res2[:,1]
#fig,sub=graf(res2,cas,2,133,p1,p2,r'Različne koncentracije jodidnih ionov na začetku')
#sub.set_title('$I^{-}(0)$=%d, $\lambda$ = 2000'%x0[0])
#############################################################


x0 = np.array([1,0,10])
cas = np.linspace(0,5,100)
p1,p2 = 1,1
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
fig, sub = graf_ura(res1,cas,3,121,p1,p2,'$\lambda$ =%.2f'%(p2/p1))
p1,p2 = 1,100
sub.set_title('strmina skoka ura je odvisna od razmerja hitrosti reakcij')
sub.grid()
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,121,p1,p2,'$\lambda$ =%.2f'%(p2/p1))

p1,p2 = 1,1000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,121,p1,p2,'$\lambda$ =%.2f'%(p2/p1))

p1,p2 = 1,5000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,121,p1,p2,'$\lambda$ =%.2f'%(p2/p1))

########################################################################

x0 = np.array([1,0,5])
cas = np.linspace(0,5,100)
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))
x0 = np.array([5,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))

x0 = np.array([10,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))
x0 = np.array([15,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))
x0 = np.array([20,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))

x0 = np.array([30,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))
x0 = np.array([35,0,5])
p1,p2 = 1,2000
res1 = odeint(model,x0,cas,args=(p1,p2))
a=res1[:,1]
fig,sub=graf_ura(res1,cas,3,122,p1,p2,'$[S_2O_3^{2-}(0)]=%d$'%(x0[2]))
sub.set_title('karakteristični čas rekacije je odvisen od začetne koncentracije $[S_2O_3^{2-}]$')

