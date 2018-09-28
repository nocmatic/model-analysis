# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:53:46 2017

@author: jure
"""
from mpl_toolkits.mplot3d import Axes3D
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
import scipy as scipy
import scipy.special as spec
import scipy.stats as stats
import timeit
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))




def gausskum(x,size=1):
    return 0.5*(1+spec.erf(x/np.sqrt(2)))

def graf1(r,phi,i,j,xlab,ylab,*args):    
    fig=plt.figure(i)
    if args:
        fig.suptitle('%s'%args[0])    
    sub= fig.add_subplot(j)
    
    plt.hist2d(r*np.cos(phi),r*np.sin(phi),bins=40,cmap = 'jet')
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab) 

    plt.colorbar(label='Število parov v razredih')
    plt.legend()
    return fig,sub

def krog(u1,u2):
    r = u1
    phi = 2* np.pi*u2
    return r,phi 


generator= np.random.RandomState()
plt.figure(1)

n = 100000
r1 = generator.rand(n)

n1 = generator.rand(n)
n2 = generator.rand(n)




plt.figure(2)
#####################################################################################
n = 10000
r1 = generator.rand(n)

n1 = generator.rand(n)
n2 = generator.rand(n)
r,phi = krog(n1,n2)


fig,sub = graf1(r,phi,1,111,'x','y', )
sub.set_title(' N=100000')

fig.tight_layout()
fig.subplots_adjust(top=0.9)
########################################################################################################
    
def krogla(u1,u2,u3):
    return u1
#######################################################

n = 100000
r1 = generator.rand(n)

n1 = generator.rand(n)
n2 = generator.rand(n)
fi = 2*np.pi*n2
def theta_solve(x,u):
    return u - 1/float(4) * ( 3*np.cos(x) - (np.cos(x))**3) - 1/float(2)
theta = np.array([])
for i in n1 : 
    thet = scipy.optimize.fsolve(theta_solve,0.0001,args=(i))
    
    theta = np.append(theta,thet)
fig = plt.figure(4)
sub = fig.add_subplot(121)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\frac{dP}{d \theta}$')

plt.hist(theta,bins=50,normed=True)
x = np.linspace(0 ,  np.pi ,100)
def dPdTheta(theta):
    return 3/float(4) * (np.sin(theta))**3
def  dPdFi(fi):
    return [1/float(2*np.pi)]*len(fi)

plt.plot(x,dPdTheta(x),'r-', label=r'$\frac{dP}{d \theta} = \frac{3 sin^3 ( \theta) }{4}$')
plt.legend()
plt.grid()
plt.title('dipolna porazdelitev po azimutalnem kotu' )
sub = fig.add_subplot(122)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\frac{dP}{d \phi}$')
x = np.linspace(0 ,  2*np.pi ,100)
plt.plot(x,dPdFi(x),'r-', label=r'$\frac{dP}{d \phi} =\frac{1}{2 \pi}$')
plt.hist(fi,bins=50,normed=True)
plt.title('dipolna porazdelitev po radialnem kotu')
plt.legend()

def graf1(theta,fi,i,xlab,ylab):      
    fig=plt.figure(i)
    ax1 = fig.add_subplot(111, projection='3d')	
    #xyz = stevila_na_sferi(N)
    x= 1* np.cos(fi)* np.sin(theta)
    y = 1* np.sin(fi)* np.sin(theta)
    z= 1 * np.cos(fi)
    ax1.scatter(x,y,z)

    #krogla
    phi1= np.linspace(0,2*np.pi,30)
    theta1 = np.linspace(0,np.pi,30)
    phi1, theta1 = np.meshgrid(phi1,theta1)
    x = np.sin(phi1)*np.cos(theta1)
    y = np.sin(phi1)*np.sin(theta1)
    z = np.cos(phi1)
    ax1.plot_surface(
        x, y, z,  rstride=1, cstride=1,alpha=0.3, color='c', linewidth=0, zorder=-1)
    ax1.axis([1,-1,1,-1])
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab) 


graf1(theta,fi,1,'x','y')
    
