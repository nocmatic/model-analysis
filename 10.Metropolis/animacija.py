# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:34:13 2017

@author: matic
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def animate(i, z, line):
    z = np.sin(x+y+i)
    print (z)
    ax.clear()
    line = ax.plot_surface(x, y, z,color= 'b')
    return line,

n = 2.*np.pi
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0,n,100)
y = np.linspace(0,n,100)
x,y = np.meshgrid(x,y)
z = np.sin(x+y)
t = np.arange(0,100)
line = ax.plot_surface(x, y, z,color= 'b')
cas = 100
ani = animation.FuncAnimation(fig, animate, cas, fargs=(z, line), interval=10, blit=False)

plt.show()