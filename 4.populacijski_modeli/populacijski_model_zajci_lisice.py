# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:52:42 2017

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
plt.rc('figure', figsize = (12,5))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'g', 'y','c', 'm', 'k'])))
#plt.cm.jet(np.linspace(0, 1, num_plots)


def graf(res,t,i,j,p,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(t,res[:,0],'b')
    b,=sub.plot(t,res[:,1,],'r')
    #c,=sub.plot(t,res[:,2],'g')
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    sub.legend([extra1, a,b],['p=%f'%p,'zajci','lisice'])
  

    sub.grid()
    
    plt.xlabel('cas[s]')
    plt.ylabel('stevilo')
    return fig
def graf1(res,t,i,j,p,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    a,=sub.plot(t,res[:,0],'b')
    b,=sub.plot(t,res[:,1,],'r')
    #c,=sub.plot(t,res[:,2],'g')
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    sub.legend([extra1, a,b],['x0=[%.2f z,%.2f l]'%(p[0],p[1]),'zajci','lisice'])
    sub.set_title('%.2f zajcev %.2f lisic'%(p[0],p[1]))

    sub.grid()
    
    plt.xlabel('cas[s]')
    plt.ylabel('stevilo')
    return fig

def zajci(x, t,p):
    '''x[0] so zajci, x[1] so lisice'''
    return np*x[0]*(1-x[1]), 1/p * x[1] *(x[0]-1)

def lot_vol(y, t, k):
    z, l = y
    dmy_z = k*z*(1-l)
    dmy_l = -1/k*l*(1-z)
    return dmy_z, dmy_l
########################################################################
'''razlicni parametri p oziroma razlicna razmerja rojevanja zajcev in umiranja lisic'''
#x0 = np.array([10,1])
#t = np.linspace(0,20,1000)
#res = odeint(zajci,x0, t,args=(1,))
#graf(res,t,1,131,1)
#
#
#x0 = np.array([10,1])
#t = np.linspace(0,25,1000)
#res = odeint(zajci,x0, t,args=(0.6,))
#graf(res,t,1,132,0.6,r'razlicno razmerje parametra p = $\sqrt{ \alpha / \beta}$')
#
#x0 = np.array([10,1])
#t = np.linspace(0,22.5,1000)
#res = odeint(zajci,x0, t,args=(1.5,))
#c=graf(res,t,1,133,1.5)
#
#c.tight_layout()
#c.subplots_adjust(top=0.90)
###################################################################################
#
#x0 = np.array([10,1])
#t = np.linspace(0,100,1000)
#res = odeint(zajci,x0, t,args=(1,))
#graf(res,t,2,131,1)
#
#
#x0 = np.array([10,1])
#t = np.linspace(0,100,1000)
#res = odeint(zajci,x0, t,args=(0.6,))
#graf(res,t,2,132,0.6,r'oscilacije populacij zajcev in lisic')
#
#x0 = np.array([10,1])
#t = np.linspace(0,100,1000)
#res = odeint(zajci,x0, t,args=(1.5,))
#d=graf(res,t,2,133,1.5)
#
#d.tight_layout()
#d.subplots_adjust(top=0.90)
#
#############################################################################
#
#x0 = np.array([10,1])
#t = np.linspace(0,100,1000)
#res = odeint(zajci,x0, t,args=(0.5,))
#graf(res,t,3,111,0.6,r'razmerje pod 0.6 naredi nezvezno in nefizikalno resitev ..')

###############################################################################
'''fazni diagram'''
def fazni_graf(res,t,i,j,p,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])
    
    sub= fig.add_subplot(j)
    sub.scatter(res[:,0],res[:,1],s=1.5)
   
  
    sub.grid()
    
    plt.xlabel('zajci')
    plt.ylabel('lisice')
    return fig,sub
#
#x0 = np.array([10,1])
#t = np.linspace(0,20,1000)
#res = odeint(zajci,x0, t,args=(1,))
#fazni_graf(res,t,4,121,1)
#
#x0 = np.array([10,1])
#t = np.linspace(0,20,1000)
#res = odeint(zajci,x0, t,args=(1.5,))
#fazni_graf(res,t,4,121,1.5)
#
#x0 = np.array([0,10])
#t = np.linspace(0,20,1000)
#res = odeint(zajci,x0, t,args=(10,))
#fazni_graf(res,t,4,121,1.5)
#
#######################################################################################
#x0 = np.array([0.00001,0.000001])
#t = np.linspace(0,50,1000)
#res = odeint(zajci,x0, t,args=(1,))
#graf(res,t,4,122,1)
#
#x0 = np.array([0,0])
#t = np.linspace(0,50,1000)
#res = odeint(zajci,x0, t,args=(1.5,))
#fazni_graf(res,t,4,122,1.5)
#
#x0 = np.array([0,0])
#t = np.linspace(0,50,1000)
#res = odeint(zajci,x0, t,args=(10,))
#fazni_graf(res,t,4,122,1.5)
#######################################################################################


'''vecanjew zajcev'''


x0 = np.array([0.5,1])
t = np.linspace(0,30,1000)
res = odeint(lot_vol,x0, t,args=(1.3,))
graf1(res,t,5,131,(0.5,1))


x0 = np.array([0.5,5])
t = np.linspace(0,30,1000)
res = odeint(lot_vol,x0, t,args=(1.3,))
graf1(res,t,5,132,(0.5,5),r'')

x0 = np.array([0.5,10])
t = np.linspace(0,30,1000)
res = odeint(lot_vol,x0, t,args=(1.3,))
c=graf1(res,t,5,133,(0.5,10),'razlicne obhodbe dobe za večanja zacetnega stevila lisic, p =1.3')

c.tight_layout()
c.subplots_adjust(top=0.90)















################################################
##'''fazni diagram'''
#def fazni_graf(res,t,i,j,p,*args):
#    
#    fig=plt.figure(i)
#    if args:
#        fig.suptitle('%s'%args[0])
#    
#    sub= fig.add_subplot(j)
##    sub.scatter(res[:,0],res[:,1],s=5)
#    a=sub.scatter(res[:,0][1:],res[:,1][1:],s=5)
#   
#    
#    sub.grid()
#      
#    plt.xlabel('Zajci')
#    plt.ylabel('Lisice')
#
#    return fig
#
#def fazni_graf1(res,t,i,j,p,*args):
#    
#    fig=plt.figure(i)
#   
#    
#    sub= fig.add_subplot(j)
#    sub.scatter(res[:,0],res[:,1],s=5)
#    if args:
#        fig.suptitle('%s'%args[0])
#        sub.set_title('resitev okoli (0,0) => (0.001, 0.001)')
#    else:   
#        sub.set_title('tocka (0,0)')
#    
#    
#    sub.grid()
#   
#    plt.xlabel('Zajci')
#    plt.ylabel('Lisice')
#
#    return fig
#
#
#
###'''ZASTOJNE TOČKE'''
###
#def zajci1(x, t,p): 
#    '''okoli prve stac. točke (0,0)'''
#    '''x[0] so zajci, x[1] so lisice'''
#    return np.array([p*x[0], -1/p * x[1]])
#
#def zajci2(x, t,p): 
#    '''okoli druge stac.točke (1,1)'''
#    '''x[0] so zajci, x[1] so lisice'''
#    return np.array([-p*x[1], 1/p * x[0]])
#
#'''stabilnost zastojnih točk'''
#x0 = [0.0,0.0]
#t = np.linspace(0,200,50)
#res = odeint(zajci1,x0, t,args=(10,))
#fazni_graf1(res,t,5,131,x0)
#
#plt.plot(0,0,color='gray')
#
#x0 = [0.001,0.001]
#t = np.linspace(0,200,50)
#res = odeint(zajci1,x0, t,args=(10,))
#b=fazni_graf1(res,t,5,132,x0,'Stabilnost zastojnih točk')
#
#
#
#fig=plt.figure(5)
#x0 = [0,0]
#t = np.linspace(0,100, 50)
#res4 = odeint(zajci2,x0, t,args=(10,))
#res4=res4+1
#
#
#x0 = [0.1,0.1]
#t = np.linspace(0,100,50)
#res5 = odeint(zajci2,x0, t,args=(10,))
#x0 = [0.2,0.2]
#t = np.linspace(0,100,50)
#res6 = odeint(zajci2,x0, t,args=(10,))
#
#    
#sub= fig.add_subplot(133)
##    sub.scatter(res[:,0],res[:,1],s=5)
#a=sub.scatter(res4[:,0],res4[:,1],s=5)
#c=sub.scatter(res5[:,0]+1,res5[:,1]+1,s=5)
#d=sub.scatter(res6[:,0]+1,res6[:,1]+1,s=5)
#plt.axhline(1,color='black',ls='--',alpha=0.5)
#plt.axvline(1,color='black',ls='--',alpha=0.5)
#sub.set_title('resitev okoli (1,1) ')
#sub.legend([a,c,d],['x0=[0,0]', 'x0= [0.1, 1]', 'x0=[0.2,0.2]'])
#  
#sub.grid()
#      
#plt.xlabel('Zajci')
#plt.ylabel('Lisice')
#
#
#b.tight_layout()
#b.subplots_adjust(top=0.80)
########################################################################################
#
#
#x0 = [0.0001,0.0001]
#t = np.linspace(0,200,50)
#res = odeint(lot_vol,x0, t,args=(10,))
#f=fazni_graf(res,t,6,121,x0,'Prava resitev')
#
#
#x0 = [0.000,0.000]
#t = np.linspace(0,200,50)
#res = odeint(lot_vol,x0, t,args=(10,))
#f=fazni_graf(res,t,6,121,x0,'Prava resitev')
#
#f[1].set_title('prava resitev okoli tocke (0,0)')
# 
#x0 = [1.1,1.1]
#t = np.linspace(0,100, 1000)
#res1 = odeint(lot_vol,x0, t,args=(10,))
#f=fazni_graf(res1,t,6,122,1.5)
#
#
#x0 = [1.5,1.5]
#t = np.linspace(0,100, 1000)
#res1 = odeint(lot_vol,x0, t,args=(10,))
#d=fazni_graf(res1,t,6,122,1.5)
#d[1].set_title('prava resitev okoli tocke (1,1)')
#
#x0 = [1,1]
#t = np.linspace(0,100,1000)
#res = odeint(lot_vol,x0, t,args=(10,))
#c=fazni_graf(res,t,6,122,1.5)
#c[0].tight_layout()
#c[0].subplots_adjust(top=0.90)
#plt.grid()