# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:29:27 2017

@author: matic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import optimize
from scipy.optimize import curve_fit
from numpy.linalg import solve
from scipy.optimize import minimize
from cycler import cycler
import scipy.special as spec
import scipy.stats as stats
import timeit
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (14,8))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['darkblue', 'lightgreen', 'darkred', 'y','c', 'm', 'k'])))

vzorec = [80,99]
populacija =[97,165]
bar_width = 0.35
opacity = 0.8
objects = ('Ruralno okolje','Urbano okolje')
tick = [1,2]
tick2 = [1  + bar_width, 2 + bar_width]
tick3= [1.1,2.1]
plt.bar(tick,vzorec, bar_width,alpha=0.5, label='vzorec',color='b')
plt.bar(tick2,populacija, bar_width,alpha=0.5,label='populacija',color='g')
plt.xticks(tick3, objects)
plt.ylabel('N')
plt.title('Porazdelitev bivanjskega prostora vzorca in populacije')
plt.legend()