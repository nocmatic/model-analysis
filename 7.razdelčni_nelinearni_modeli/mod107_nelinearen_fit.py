# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:42:31 2017

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




from scipy.optimize import least_squares
'''u[0] = y0, u[1] = a, u[2] = p'''
def model(x, u):
    return u[0]*x**u[2]/ (x**u[2] + u[1]**u[2])

def fun(u, x, y,yerr):
     return (model(x, u) - y)/yerr
 
def jac(u, x, y,yerr):
     J = np.empty((x.size, u.size))
    
     J[:, 0 ] =x**u[2]/ (x**u[2] + u[1]**u[2])
     J[:, 1] =  u[0]*x**u[2]/ (x**u[2] + u[1]**u[2])**2 * (-1)* u[2]*u[1]**(u[2]-1)
     J[:, 2] =  u[0]*x**u[2] * u[1]**u[2] * (np.log(x) - np.log(u[1])) /  (x**u[2] + u[1]**u[2])**2
     return J
 



##############################################################################################
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

podatki=np.loadtxt('farmakoloski.dat',skiprows=1)
x = np.array(podatki[:,0])
y = np.array(podatki[:,1])


yerr =3

u0 = np.array([104.683073,21.189348,1])
res = least_squares(fun, u0, method='lm', args=(x, y,yerr))
print('$2.chi^2 $= %.3f'%(res.cost))
print('$chi_{reduced}^2 $= %.3f'%(res.cost/(len(x)-3)))
print('res:' , res.x)
y0,a,p= res.x
H = np.linalg.inv(res.jac.T.dot(res.jac))
var_test = np.sqrt(np.diagonal(H))
plt.figure(2)
plt.xlabel('Vnos reagenta X')
plt.ylabel('Odziv tkiva Y')
x1 = np.linspace(0,x[-1],100)
plt.plot(x1, model(x1,[y0,a,p]),'r',label='fit')
plt.plot(x,y,'ko',label='meritve')
plt.grid()
plt.errorbar(x,y,yerr,fmt='ko')
plt.legend()
def fun(x,y0,a,p):
     return y0*x**p/ (x**p + a**p)



yerr=np.array(len(y)*[3])

popt, pcov = curve_fit(fun, x, y, u0, sigma = yerr,method='lm')


var = np.sqrt(np.diagonal(pcov))
def chi3(fun,x,y,yerr,popt):
    y0,a,p =popt
    
    chi= sum((fun(x,y0,a,p)-y)**2/yerr**2)
        
    return chi
chi = chi3(fun,x,y,yerr,popt)
chi_r = chi / (len(x)-3-1)
def korelacije(pcov, var):
    korelacije=[pcov[0][1],pcov[0][2],pcov[1][2]]
    
    korelacijski = [ korelacije[0] /(var[0]*var[1]) , korelacije[1] / (var[0]*var[2]), korelacije[2] / (var[1]*var[2]) ]
    return korelacijski
a = korelacije(pcov,var)
print (popt)
plt.text(400,40,r'korelacijski koeficienti: $ \rho_{y_0,a}$ = %.3f,$ \rho_{y0,p}$= %.3f,$ \rho_{a,p}$ = %.3f'%(a[0],a[1],a[2]))

plt.text(400,60,' fit : $y = y_0 x^p/(x^p + a^p)$ \n $y_{err}=%.3f$ \n $chi^2 $= %.3f \n $chi_{reduced}^2 $= %.3f \n $y_0$ = %.3f $\pm$ %.3f \n a = %.3f $\pm$ %.3f \n p %.3f $\pm$ %.3f'%(yerr[0], chi,chi_r,res.x[0],var[0],res.x[1],var[1],res.x[2],var[2]))
def chi(fun,x,y,yerr,popt):
    y0,a,p =popt
    
    chi= (fun(x,y0,a,p)-y)**2/yerr**2
        
    return chi
plt.figure(3)
plt.xlabel('napake')
plt.ylabel('$chi^2_i$')

plt.plot(x, chi(fun,x,y,yerr,popt),'r+',label='napake', )

plt.plot(x, chi(fun,x,y,yerr,popt),'r--',label='napake',alpha =0.3 )
plt.grid()


plt.title('napaka fita')

chi1 = chi(fun,x,y,yerr,popt)
chi2 = chi3(fun,x,y,yerr,popt)
print (chi2/(len(x)-4))