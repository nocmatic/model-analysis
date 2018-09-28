# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:42:39 2017

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
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))

'''x[0] je snov H2, x[1] je snov Br2, x[2] je snov HBr'''

def graf(res,cas,i,j,k,m,*args):  
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(cas,res[:,0],'darkblue')
    b,=sub.plot(cas,res[:,1],'lightblue')
    c,=sub.plot(cas,res[:,2],'violet')
    k1=res[0,0]/res[0,1]
    
#    sub.set_title(r'$[HBr(0)]= %d$'%(res[0,2]))
    sub.set_title(r'$\frac{[H_2(0)]}{[Br_2(0)]}= %0.2f$'%(k1))
    sub.legend([a,b,c],['$H_2$','$Br_2$','$HBr$'],loc=0)
    
  
    sub.grid()    
    plt.xlabel('$čas[s]$')
    plt.ylabel('$koncentracija$')
#    return fig,sub
def model(x,cas,k,m):
    
    return np.array([-k*x[1]**(1/2)*x[0]/(x[2]/x[1] + m),
                     -2*k*x[0]*x[1]**(1/2)/(x[2]/x[1] + m),
                     2*k*x[0]*x[1]**(1/2)/(x[2]/x[1] + m)])
    
'''Grafi'''
x0 = np.array([100,1,0])
cas = np.linspace(0,9,1000)
k,m = 1,50
res1 = odeint(model,x0,cas,args=(k,m))
a=res1[:,1]
graf(res1,cas,1,131,k,m)

x0 = np.array([1,1,0])
cas = np.linspace(0,100,1000)
k,m = 1,50
res2 = odeint(model,x0,cas,args=(k,m))
b=res2[:,1]
graf(res2,cas,1,132,k,m,r'$Časovna$ $odvisnost$ $koncentracij$ $za$ $različne$ $[H_2(0)]$ $in$ $[Br_2(0)]$ $pri$ $m=50$, $k=1$')

x0 = np.array([1,100,0])
cas = np.linspace(0,9,1000)
k,m = 1,50
res3 = odeint(model,x0,cas,args=(k,m))
c=res3[:,1]
#fig,sub = graf(res3,cas,1,133,k,m)


#sub.set_title
#################################################################################################
'''razmerje 100, HBr(0)=10 in 100'''
#x0 = np.array([100,1,10])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res1 = odeint(model,x0,cas,args=(k,m))
#a=res1[:,1]
#graf(res1,cas,2,131,k,m)
#
#x0 = np.array([100,1,100])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res2 = odeint(model,x0,cas,args=(k,m))
#graf(res2,cas,2,132,k,m,r'$Časovna$ $odvisnost$ $koncentracij$ $za$ $\frac{[H_2(0)]}{[Br_2(0)]}=100$ $pri$ $m=2.5$, $k=1$')
#
#x0 = np.array([100,1,150])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res4 = odeint(model,x0,cas,args=(k,m))
#c=res4[:,1]
#graf(res4,cas,2,133,k,m)

'''razmerje 1, HBr(0)=10 in 100'''
#x0 = np.array([1,1,10])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res2 = odeint(model,x0,cas,args=(k,m))
#b=res2[:,1]
#graf(res2,cas,3,131,k,m,r'$Časovna$ $odvisnost$ $koncentracij$ $za$ $\frac{[H_2(0)]}{[Br_2(0)]}=1$ $pri$ $m=2.5$, $k=1$')
#
#x0 = np.array([1,1,100])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res3 = odeint(model,x0,cas,args=(k,m))
#c=res3[:,1]
#graf(res3,cas,3,132,k,m)
#
#x0 = np.array([1,1,150])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res4 = odeint(model,x0,cas,args=(k,m))
#c=res4[:,1]
#graf(res4,cas,3,133,k,m)

'''razmerje 0.01, HBr(0)=10 in 100'''
#x0 = np.array([1,100,10])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res2 = odeint(model,x0,cas,args=(k,m))
#b=res2[:,1]
#graf(res2,cas,4,131,k,m,r'$Časovna$ $odvisnost$ $koncentracij$ $za$ $\frac{[H_2(0)]}{[Br_2(0)]}=0.01$ $pri$ $m=2.5$, $k=1$')
#
#x0 = np.array([1,100,100])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res3 = odeint(model,x0,cas,args=(k,m))
#c=res3[:,1]
#graf(res3,cas,4,132,k,m)
#
#x0 = np.array([1,100,150])
#cas = np.linspace(0,8,1000)
#k,m = 1,2.5
#res4 = odeint(model,x0,cas,args=(k,m))
#c=res4[:,1]
#graf(res4,cas,4,133,k,m)
    
