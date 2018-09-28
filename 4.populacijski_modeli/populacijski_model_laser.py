# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:54:33 2017

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
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (12,4))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'g', 'y','c', 'm', 'k'])))
#plt.cm.jet(np.linspace(0, 1, num_plots)


def graf(res,t,i,j,p,q,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(t,res[:,0],'b')
    b,=sub.plot(t,res[:,1,],'r')
    #c,=sub.plot(t,res[:,2],'g')
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    sub.legend([extra1, a,b],['p=%f  q=%f'%(p,q),'atomi','fotoni'])
  

    sub.grid()
    
    plt.xlabel('cas[s]')
    plt.ylabel('stevilo')
    return fig,sub

def laser(x, t,p,q):
    '''x[0] so atomi, x[1] so fotoni'''
    return np.array([q - p* x[0] * (x[1] +1), x[1]/p * (x[0]-1)])

def fazni_graf(res,t,i,j,p,q,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    sub.scatter(res[:,0][0],res[:,1][0],s=10, color='r')
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    sub.legend([extra1],['p=%f  q=%f'%(p,q)])
   
    sub.plot(res[:,0],res[:,1],'r')
  
    sub.grid()
    
    plt.xlabel('atomi')
    plt.ylabel('fotoni')
    return fig,sub


########################################################################
'''stacionarna tocka (1,0)'''
x0 = np.array([200,200])
t = np.linspace(0,200,1000)
res = odeint(laser,x0, t,args=(1,10))
fazni_graf(res,t,2,121,1,1)

array=[]
x0 = np.array([0.5,0.5])
t = np.linspace(0,60,1000)
p,q = 0.1,1
res = odeint(laser,x0, t,args=(p,q))
fig,sub=graf(res,t,5,131,p,q,'Variranje črpanja za odvisnost oscilacij')
sub.set_title('')

x0 = np.array([0.5,0.5])
t = np.linspace(0,60,1000)
p,q = 0.1,1.5
res = odeint(laser,x0, t,args=(p,q))
fig,sub=graf(res,t,5,132,p,q)
sub.set_title('')
#fig.subplots_adjust(top=0.90)
#
#fig.tight_layout()
#sub.set_title('manjše napajanje')
#
x0 = np.array([0.5,0.5])
t = np.linspace(0,60,1000)
p,q = 0.1,3
res = odeint(laser,x0, t,args=(p,q))
fig3,sub=graf(res,t,5,133,p,q)
sub.set_title('')
fig3.tight_layout()

fig3.subplots_adjust(top=0.90)
##################################################################################
#
#x0 = np.array([50,50])
#t = np.linspace(0,2000,1000)
#res = odeint(laser,x0, t,args=(1,1.2))
#fazni_graf(res,t,3,121,1,1)
#
#
#x0 = np.array([50,50])
#t = np.linspace(0,2000,1000)
#res = odeint(laser,x0, t,args=(1,1.2))
#a= graf(res,t,3,122,1.2,1.2,'blizu tocke 1,1 je še vedno stabilna resitev')
#a.tight_layout()
#a.subplots_adjust(top=0.90)


###########################################################################################
#'''zastojne točke'''
#x0 = np.array([1,0])
#t = np.linspace(0,60,1000)
#p,q = 1,1
#res = odeint(laser,x0, t,args=(p,q))
#fig,sub=fazni_graf(res,t,3,121,p,q,'Zastojna točka (a, f) = (q/p, 0)')
#sub.set_title('zastojna točka (q/p , 0) = (1,0)')
#
#x0 = np.array([1.001,0.001])
#t = np.linspace(0,200,1000)
#p,q = 1,1
#res = odeint(laser,x0, t,args=(p,q))
#fig,sub=fazni_graf(res,t,3,122,p,q,)
#sub.set_title('mali odmik iz zastojne točke (q/p , 0) ')

#"############################################################################################
x0 = np.array([1,0.5])
t = np.linspace(0,60,1000)
p,q = 1,1.5
res = odeint(laser,x0, t,args=(p,q))
fig,sub=fazni_graf(res,t,3,121,p,q,'Zastojna točka (a, f) = (1, q/p-1)')
sub.set_title('zastojna točka (1, q/p -1) = (1,0.5)')

x0 = np.array([1.00,0.6])
t = np.linspace(0,200,1000)
p,q = 1,1.5
res = odeint(laser,x0, t,args=(p,q))
fig,sub=fazni_graf(res,t,3,122,p,q,)
sub.set_title('mali odmik iz zastojne točke (1, q/p -1) ')
##
#x0 = np.array([0.5,0.5])
#t = np.linspace(0,60,1000)
#p,q = 0.1,1.5
#res = odeint(laser,x0, t,args=(p,q))
#fig,sub=fazni_graf(res,t,3,132,p,q)
#sub.set_title('')
##fig.subplots_adjust(top=0.90)
##
##fig.tight_layout()
##sub.set_title('manjše napajanje')
##
#x0 = np.array([0.5,0.5])
#t = np.linspace(0,60,1000)
#p,q = 0.1,3
#res = odeint(laser,x0, t,args=(p,q))
#fig1,sub=fazni_graf(res,t,3,133,p,q)
#sub.set_title('')
#fig1.tight_layout()
#
#fig1.subplots_adjust(top=0.90)














