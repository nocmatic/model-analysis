# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:13:01 2018

@author: matic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math

from scipy.optimize import curve_fit
import time
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import fft
import scipy as scipy
from scipy import optimize
from cycler import cycler
plt.rc('text', usetex = False)
plt.rc('font', size = 15, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (17,7))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))


def Tdot(T,t, x, A, v,x_stacionarna, b,kse,kpe):
    
    '''to je odvod sile po času, spremenljivke:
            x = x(t) - odmik mišice v odvisnosti od časa
            x_stacionarna - ravnovesni položaj mišice
            v = v(t) - hitrost razmikanja mišice 
            b, kse,kpe - parametri vzmeti in elementov'''
            
    ''' PAZI : konstante so x_stac, b, kse,kpe'''
    ''' T, x,A,v morajo biti seznami v odvisnosti od časa, npr: x = [1, 1.1, 1.2, 1.3, 1.4, 1.5,....] '''
    
    return kse/b * (kse * (x(t) - x_stacionarna) + b*v(t) - (1 + kpe/kse)*T + A(t))

def x(t):
    '''primer razteg misice kot v članku'''
    razteg = []
    
    for i in t:
        if i <= 0.05:
            razteg.append( (2-1)/0.05 * i + 1)
        if i > 0.05 and i <=1:
            razteg.append(2)
        if i >1 and i <=1.05 :
           razteg.append( (1-2)/0.05 * i + 22)
           
        if i > 1.05:
           razteg.append(1)
    return razteg
def x_simulacija1(t):
 
    if t<= 0.05:
          return (2-1)/0.05 * t + 1
    if t > 0.05 and t <=1:
           return 2
    if t >1 and t <=1.05 :
            return ((1-2)/0.05 * t + 22)
           
    if t > 1.05:
           return 1
    
def v_simulacija1(t):
    if t <= 0.05:
          return (2-1)/0.05 
    if t > 0.05 and t <=1:
           return 0
    if t >1 and t <=1.05 :
            return ((1-2)/0.05)
           
    if t > 1.05:
           return 0
    

def v(t,x):
    '''daš notri array raztezka x(t) in ti izračuna gradient (odvod), potem moraš funkcijo sam sestavit'''
    return np.gradient(x)
def A_konst(t,A0=0):
    '''konstantno vzbujanje'''
    return A0

#%%
'''poskusimo rešiti primer ki je v članku'''
t = np.linspace(0,2,200)

razteg = x(t)
hitrost = v(t,razteg)
indeksi = np.arange(200)

fig = plt.figure(1)
sub = fig.add_subplot(121)
sub.plot(t,razteg, 'k')
sub.set_title('x(t)')
sub.set_xlabel('cas [s]')
sub.set_ylabel('x [cm]')
plt.grid()

sub2= fig.add_subplot(122)
sub2.plot(t,hitrost,'r')
sub2.set_title('v(t)')
sub2.set_xlabel('cas [s]')
sub2.set_ylabel('v [cm/s]')
fig.tight_layout()
plt.grid()
#####################################################################
'''sedaj rešimo diferencialko, poklicemo funckijo odeint'''
'''https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html'''
''' x, A, v,x_stacionarna, b,kse,kpe'''
t = np.linspace(0,2,400)
b = 50
kse= 125
kpe= 75
A =  1
x_stacionarna=1
y0 = [5]
T = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))


fig = plt.figure(3)
plt.title('Odvisnost sile od zaporedne vzmeti')
sub = fig.add_subplot(111)
sub.set_xlabel('cas [s]')
sub.set_ylabel('T [N]')
plt.plot([],[],' ',label=r'$K_{SE}= %.2f, b = %.2f, A = %.2f$'%(kpe,b,A))
a,= plt.plot(t,T,'r', label=r'$K_{SE} = %.2f$'%kse)
kse= 100
T2 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T2,'g', label=r'$K_{SE} = %.2f$'%kse)
kse= 75
T3 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T3,'b', label=r'$K_{SE} = %.2f$'%kse)
kse= 30
T4 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T4,'c', label=r'$K_{SE} = %.2f$'%kse)
plt.legend(loc='best')

plt.grid()



#plt.text(1.25, 50, r'$K_{SE} = %.2f$'%kse +'\n'+ r'$K_{PE} = %.2f$'%kpe + '\n'+ r'$ b = %.2f$ '%b, 
#        bbox={'facecolor': 'white', 'alpha':0.5, 'pad':10})

plt.title('Odvisnost T od casa pri maksimalnem stimulusu (ko postane sila A konstantna oziroma saturirana)')
#%%

t = np.linspace(0,2,400)
b = 50
kse= 135
kpe= 75
A =  1
x_stacionarna=1
y0 = [5]
T = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))


fig = plt.figure(4)
plt.title('Odvisnost sile od paralelne vzmeti')
sub = fig.add_subplot(111)
sub.set_xlabel('cas [s]')
sub.set_ylabel('T [N]')
plt.plot([],[],' ',label=r'$K_{SE}= %.2f, b = %.2f, A = %.2f$'%(kse,b,A))
a,= plt.plot(t,T,'r', label=r'$K_{PE} = %.2f$'%kpe)
kpe= 40
T2 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T2,'g', label=r'$K_{PE} = %.2f$'%kpe)
kpe= 20
T3 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T3,'b', label=r'$K_{PE} = %.2f$'%kpe)
kpe= 5
T4 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T4,'c', label=r'$K_{PE} = %.2f$'%kpe)
plt.legend(loc='best')
plt.grid()


t = np.linspace(0,2,400)
b = 50
kse= 135
kpe= 75
A =  1
x_stacionarna=1
y0 = [5]
T = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))


fig = plt.figure(5)
plt.title('Odvisnost sile od dušenja')
sub = fig.add_subplot(111)
sub.set_xlabel('cas [s]')
sub.set_ylabel('T [N]')
plt.plot([],[],' ',label=r'$K_{SE}= %.2f, K_{PE} = %.2f, A = %.2f$'%(kse,kpe,A))
a,= plt.plot(t,T,'r', label=r'$b = %.2f$'%b)
b=150
T2 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T2,'g', label=r'$b = %.2f$'%b)
b= 10
T3 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T3,'b', label=r'$b = %.2f$'%b)
b=0.001
T4 = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija1, A_konst, v_simulacija1, x_stacionarna, b , kse, kpe))
a,= plt.plot(t,T4,'k', label=r'$b = %.2f$'%b)
plt.legend(loc='best')
plt.grid()

#%%
def twitch(t):
    return 4950*(np.exp(-t/0.034)- np.exp(-t/0.0326))
def impulse(t):
    return 48144*(np.exp(-t/0.0326)) - 45845*np.exp(-t/0.034)
fig = plt.figure(6)
sub = fig.add_subplot(121)
t = np.linspace(0,0.35,200)
plt.plot(t,twitch(t),'ko')
sub.set_title('Sunek mišice - izometrična obremenitev')
plt.grid()


sub = fig.add_subplot(122)
t = np.linspace(0,0.35,200)
plt.plot(t,impulse(t),'ko')
sub.set_title('Potek sunka aktivne sile - izometrična obremenitev ')


fig.tight_layout()
plt.grid()

#%%
'''poskusimo sedaj ali res dani sunek aktivne sile povroži takšno silo mišice'''

def x_simulacija2(t):
 
    return 6
    
def v_simulacija2(t):
    return 0
t = np.linspace(0,20,200)
b = 50
kse= 136
kpe= 75

x_stacionarna=1
y0 = [5]
T = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija2, impulse, v_simulacija2, x_stacionarna, b , kse, kpe))


fig = plt.figure(7)
plt.title('Odvisnost sile od zaporedne vzmeti')
sub = fig.add_subplot(111)
sub.set_xlabel('cas [s]')
sub.set_ylabel('T [N]')
plt.plot([],[],' ',label=r'$K_{SE}= %.2f, b = %.2f, A = %.2f$'%(kpe,b,A))
a,= plt.plot(t,T,'r', label=r'$b = %.2f$'%b)
#%%
'''sestavimo sedaj več sunkov sile'''
tk = 1
t = np.linspace(0,tk,100)

impulz = impulse(t)
st_sunkov = 100
def sestavljena_sunki(t, impulz= impulz,st_sunkov = st_sunkov,tk = tk, ):
    Deltat = tk/st_sunkov
    st_ijev = len(impulz)
    stevilo_sekanja = Deltat*st_ijev
    print(stevilo_sekanja)
    novi_impulz = np.zeros(2* len(impulz))
    novi_impulz[0:len(impulz)] = novi_impulz[0:len(impulz)] + impulz
    
    for i in range(st_sunkov):
        if i == 0 :
           continue
        else:
           
            novi_impulz[int(i*stevilo_sekanja):int(i*stevilo_sekanja+st_ijev)] =  novi_impulz[int(i*stevilo_sekanja):int(i*stevilo_sekanja+st_ijev)] + impulz
    return novi_impulz

array = sestavljena_sunki(t)
t = np.linspace(0,2*tk,180)
tk = t[-1]*2
dt = t[1]-t[0]
def sestavljena_funkcija_sunki(t, array=array, t0=0,tk=tk,dt=dt):
    i = int((t-t0)/dt)
    return array[i]



b = 50
kse= 125
kpe= 75

x_stacionarna=1
y0 = [5]
T = scipy.integrate.odeint(Tdot,y0,t, args= (x_simulacija2, sestavljena_funkcija_sunki, v_simulacija2, x_stacionarna, b , kse, kpe))
fig = plt.figure(12)
plt.title('20 Hz')
sub = fig.add_subplot(111)
sub.set_xlabel('cas [s]')
sub.set_ylabel('T [N]')
plt.plot([],[],' ',label=r'$K_{SE}= %.2f, b = %.2f, A = %.2f$'%(kpe,b,A))
a,= plt.plot(t,T,'r', label=r'$b = %.2f$'%b)
plt.grid()

#%%



















