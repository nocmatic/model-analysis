# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:31:48 2018

@author: matic
"""
import numpy as np
import timeit
import numba
from numba import jit
def id(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


a = np.zeros((10, 10)); aid = id(a); aid 
b = a.reshape((1, -1)); id(b) == aid

@jit
def test():
    new = np.array([])

    for i in range(20000):
        new = np.append(new,i)
        

def test2():
    new = np.zeros(20000)
    for i in range(20000):
        new[i] = i
    return 'a'
test3 = numba.jit(test2)
#%timeit test()

#%timeit test2()
a = np.array([1,2]); aid = id(a);


b = np.array(a); id(b) == aid


