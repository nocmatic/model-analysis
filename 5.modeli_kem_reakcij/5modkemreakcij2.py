# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:15:21 2017

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
plt.rc('font', size = 10, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (12,5))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'g', 'y','c', 'm', 'k'])))



##############################################################
'''stacionarna rešitev'''
def A1(A,p,q,r):
    return p*A*A/(q*A + r)

def kem2(x,t, p,q,r):
    
    '''prvi je dovzetni, drugi je bolan, tretji je imun'''
    
    return np.array([-p* x[0]*x[0]+ q*x[0]*A1(x[0],p,q,r), r* A1(x[0],p,q,r), r* A1(x[0],p,q,r)])
def graf_stac(res,t,i,j,p,q,r,n0,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args)
    
    sub= fig.add_subplot(j)
    a,=sub.plot(t,res[:,0],'b', label='[A]')
    d,=sub.plot(t,A1(res[:,0],p,q,r),'g', label='[A*]')
    b,=sub.plot(t,res[:,1,],'r', label='[B]')
    c,=sub.plot(t,res[:,2,],'m', label='[C]')
    
    
    array= ['[A]','[A*]','[B]','[C]' ]
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    
    sub.legend(array, loc='best')
    sub.grid()
    
    plt.xlabel('cas[s]')
    plt.ylabel('koncentracija')
    return fig,sub

n0 = 10
p = 1000
x0= np.array([10,0,0])
q = p/1000
r = q* n0/x0[0]

n0 = 0.01


r = n0*p*1000/x0[0]
t = np.linspace(0,3000,1000)
konst = (p,q,r)
res = odeint(kem2,x0, t,args=konst)
fig, sub =graf_stac(res,t,2,131,p,q,r,n0,'Stacionarna rešitev, obrnjeni p in q. A0= [10,0,0,0]' )
sub.set_title('p=%.2f q=%.2f r = %.2f*q/A(0)=%.2f'%(p,q,n0,r))

