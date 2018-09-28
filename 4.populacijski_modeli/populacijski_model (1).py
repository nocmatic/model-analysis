# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:08:50 2017

@author: Matic
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

def epidemija(x,t, alfa,beta):
    '''prvi je dovzetni, drugi je bolan, tretji je imun'''
    
    return np.array([-alfa* x[0]*x[1], alfa* x[0]*x[1] - beta*x[1], beta*x[1]])

def graf(res,t,i,j,alfa,beta,*args):
    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args)
    
    sub= fig.add_subplot(j)
    a,=sub.plot(t,res[:,0],'b')
    b,=sub.plot(t,res[:,1,],'r')
    c,=sub.plot(t,res[:,2],'g')
    extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    
    sub.legend([extra1,extra2,a,b,c], [r'$\alpha$ = %.2e'%alfa,r'$ \beta$ = %.2e'%beta,'dovzeten','bolan','imun/mrtev'])

    sub.grid()
    
    plt.xlabel('cas[s]')
    plt.ylabel('stevilo')
    return fig

'''SIMPL PRIMERI:'''

x0= np.array([500,10,0])
t = np.linspace(0,10000,1000)
res = odeint(epidemija,x0, t,args=(0.00001,0.001))
graf(res,t,1,111,0.00001,0.001)
#VRH EPIDEMIJE RAZIŠČI
##############################################################################################
'''dlje casa ker so konstantne manjse!'''
x0= np.array([500,10,0])
t = np.linspace(0,1000000,1000)
res2 = odeint(epidemija,x0, t,args=(0.00001,0.000001))
graf(res2,t,2,111,0.00001,0.000001)



################################################################################################
'''iste konstante le da ljudi pocepimo, ce jih dovolj cepimo ni več viška'''
x0= np.array([5000,10,0])
t = np.linspace(0,10000,1000)
res2 = odeint(epidemija,x0, t,args=(0.000001,0.001))
graf(res2,t,6,131,0.000001,0.001, 'Cepljenje ljudi')


t = np.linspace(0,13000,1000)
x0= np.array([2000,10,3000])
res2 = odeint(epidemija,x0, t,args=(0.000001,0.001))
graf(res2,t,6,132,0.000001,0.001)

t = np.linspace(0,20000,1000)
x0= np.array([1000,10,4000])
res2 = odeint(epidemija,x0, t,args=(0.000001,0.001))
a= graf(res2,t,6,133,0.000001,0.001)

#ZA LEPO PORAVNAVO GRAFOV!
a.tight_layout()
a.subplots_adjust(top=0.90)


##############################################################################################
'''odvisnost od hitrosti širjenja alfa je tudi višina vrha epidemije'''

x0= np.array([5000,10,0])
t = np.linspace(0,10000,1000)
res2 = odeint(epidemija,x0, t,args=(0.00001,0.001))
graf(res2,t,8,131,0.00001,0.001, 'Nizanje vrha epidemije')

res2 = odeint(epidemija,x0, t,args=(0.000001,0.001))
graf(res2,t,8,132,0.000001,0.001)

res2 = odeint(epidemija,x0, t,args=(0.0000005,0.001))
a= graf(res2,t,8,133,0.0000005,0.001)

#ZA LEPO PORAVNAVO GRAFOV!
a.tight_layout()
a.subplots_adjust(top=0.90)
res2 = odeint(epidemija,x0, t,args=(0.0000001,0.001))
graf(res2,t,5,122,0.0000001,0.001)

#############################################################################################

''''''

def epidemija1(x,t, *args):
    '''prvi je dovzetni, drugi je bolan1, tretji bolan2,...N=len(konst) je imun'''
    #diagonala
    konst1 = np.array(args)
    konst = np.append(konst1,0)
  
    a = np.zeros((len(konst)-1, len(konst)-1))
    np.fill_diagonal(a, -1*konst[1:])
    
    #spodnja diagonala 
    b = np.zeros((len(konst)-2, len(konst)-2))
    np.fill_diagonal(b, konst[1:], wrap=True)

    c = np.array([np.zeros(len(konst)-2)])
    d= np.concatenate((b, c.T), axis=1)
    
    c = np.array([np.zeros(len(konst)-1)])
    
    e =np.concatenate((c, d), axis=0)
   
    rest = np.array((a+e).dot(x[1:]))
   
    alfa= np.array([-konst[0]* x[0]*x[1] ,konst[0]* x[0]*x[1] + rest[0] ])
        
    return np.concatenate((alfa,rest[1:]),axis=0)
    

def graf1(res,t,i,j,*args):
    
    fig=plt.figure(i)
   
    
    sub= fig.add_subplot(j)
    if args:
       sub.set_title('%s'%args[0]) 
    array=['$B_{%d}$'%k for k in range(len(res.transpose()))]
    array[0] = 'Dovzetni'
    array[len(res.transpose())-1] = 'Imuni/mrtvi'
    plt.gca().set_prop_cycle(cycler('color', ['b', 'r', 'g', 'y','c', 'm', 'k']))
#plt.cm.jet(np.linspace(0, 1, num_plots)
    for i in range(len(res.transpose())):        
        sub.plot(t,res[:,i],label=array[i])   
            
    sub.legend( array,loc='best')
    sub.grid()
    plt.xlabel('cas[s]')
    plt.ylabel('stevilo')
    return fig


x0= np.array([5000,10,0])
t = np.linspace(0,5000,1000)
res2 = odeint(epidemija1,x0, t,args=(0.000001,0.001))
graf1(res2,t,7,131, 'vec stopenj - samo ena bolezen')



x0= np.array([5000,10,0,0,0])
t = np.linspace(0,8000,1000)
res2 = odeint(epidemija1,x0, t,args=(0.000001,0.001,0.001,0.001))
graf1(res2,t,7,132, 'vec stopenj - 3 stopnje bolezni ')

x0= np.array([5000,10,0,0,0,0,0])
t = np.linspace(0,8000,1000)
res2 = odeint(epidemija1,x0, t,args=(0.000001,0.001,0.001,0.001,0.001,0.001))
graf1(res2,t,7,133, 'vec stopenj - 5 stopnj bolezni ')
