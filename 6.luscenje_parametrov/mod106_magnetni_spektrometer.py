# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:24:07 2017

@author: ASUS
"""
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy import optimize
import sys  
from scipy.optimize import curve_fit
from numpy.linalg import solve
from scipy.optimize import minimize
from cycler import cycler
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))

meritve=np.loadtxt('thtg-xfp-thfp.dat')

theta_tg00=meritve[:,0]
x_fp00=meritve[:,1]
theta_fp00=meritve[:,2]

theta_tg1=theta_tg00[::10]
x_fp1=x_fp00[::10]
theta_fp1=theta_fp00[::10]

theta_tg=[]
x_fp=[]
theta_fp=[]

for i in range(len(theta_fp1)):
    theta_tg = np.append(theta_tg,theta_tg00[i*11:(i+1)*11-1])
    x_fp = np.append(x_fp,x_fp00[i*11:(i+1)*11-1])
    theta_fp = np.append(theta_fp,theta_fp00[i*11:(i+1)*11-1])

'''ločila na dve skupini, ena ima vsak stoti element iz skupine meritev, druga vse ostale'''
mrad = 0.572958 # za napako dam velikostni red 10miliradianov
z = theta_tg/mrad
x1 = theta_fp/mrad
x2 = theta_fp**2/mrad**2
y1 = x_fp
y2 = x_fp**2

A0 = np.vstack([np.ones(len(x1))/mrad, x1, x2, y1, y2]).T
resitev0 = np.linalg.lstsq(A0,z)
a0, a1, a2, b1, b2 = resitev0[0]
chi_square0 = resitev0[1]
ocena0 = resitev0[1]/(len(x1)-3) #Ocena za napako == REDUCIRAN CHI^2

M0 = A0.T.dot(A0)
U,s,V=np.linalg.svd(M0)
pinv_svd = np.dot(np.dot(V.T,np.linalg.inv(np.diag(s))),U.T)
da0,da1,da2,db1,db2 = np.sqrt((pinv_svd)[0][0]*ocena0),np.sqrt((pinv_svd)[1][1]*ocena0),np.sqrt((pinv_svd)[2][2]*ocena0),np.sqrt((pinv_svd)[3][3]*ocena0),np.sqrt((pinv_svd)[4][4]*ocena0)
print('$a_0$=%0.4f \pm %0.4f, $a_1$=%0.4f \pm %0.4f, $a_2$=%0.4f \pm %0.4f, $b_1$=%0.4f \pm %0.4f, $b_2$=%0.4f \pm %0.4f'%(a0,da0,a1,da1,a2,da2,b1,db1,b2,db2))

res = a0 + a1*x1 + a2*x2 + b1*y1 + b2*y2

#fig1=plt.figure(1)
#sub = fig1.add_subplot(121)
#sub.plot(z,res,color='lightgreen')
#plt.xlabel(r'$\theta_{tg}$')
#plt.ylabel(r'$\theta_{model}$')
#sub.grid()
#plt.title(r'$\theta_{model}(\theta_{tg})$')
#
#razlika=res-z
#sub = fig1.add_subplot(122)
#sub.plot(razlika,color='lightgreen')
#plt.xlabel(r'$N$')
#plt.ylabel(r'$\theta_{model}-\theta_{tg}$')
#sub.grid()
#plt.legend()
#plt.title(r'$Razlike$')
#
#
def chisq(y_testni,y_fit):
    chi = sum((y_testni - y_fit)**2)
    return chi

y_testni = theta_tg1/mrad
x1 = theta_fp1/mrad
x2 = theta_fp1**2/mrad**2
y1 = x_fp1
y2 = x_fp1**2

res = a0 + a1*x1 + a2*x2 + b1*y1 + b2*y2
y_fit = res

mera =chisq(y_testni,y_fit)
norm_mera=mera / len(theta_fp1)
