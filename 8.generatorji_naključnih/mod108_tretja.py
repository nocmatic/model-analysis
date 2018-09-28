# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:10:48 2017

@author: ASUS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy import stats
from cycler import cycler
from scipy.stats import cumfreq
import scipy.stats as stats
plt.rc('text', usetex = False)
plt.rc('font', size = 11, family = 'serif', serif = ['Computer Modern'])
plt.rc('xtick', labelsize = 'medium')
plt.rc('ytick', labelsize = 'medium')
plt.rc('legend', frameon = False, fontsize = 'medium')
plt.rc('figure', figsize = (16,6))
plt.rc('lines', linewidth=2.0)
plt.rc('axes', prop_cycle=(cycler('color', ['purple', 'y', 'darkred', 'y','c', 'm', 'k'])))



def funkcija1(leto,st):
    if st<=9:
        pod1 = pd.read_csv('mod_tm%s_10%s.dat'%(leto,st),sep=":",header=None)
    else:
        pod1 = pd.read_csv('mod_tm%s_1%s.dat'%(leto,st),sep=":",header=None)
    return np.array(pod1[0]*24 + pod1[1] + pod1[2]/60).T

def normalizacija(podatki):
#    x_org = podatki       #to so original podatki za ksneje
    xpovp= podatki.mean() #povprečje x
    sigmax= podatki.std() #standardni odklon
    x = (podatki-xpovp)/sigmax
    return x

def povprecje(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
    xpovp= np.array([p1.mean(),p2.mean(),p3.mean(),p4.mean(),p5.mean(),p6.mean(),p7.mean(),p8.mean(),p9.mean(),p10.mean(),p11.mean(),p12.mean(),p13.mean()]) #povprečje x
    return xpovp

def povprecje1(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12):
    xpovp= np.array([p1.mean(),p2.mean(),p3.mean(),p4.mean(),p5.mean(),p6.mean(),p7.mean(),p8.mean(),p9.mean(),p10.mean(),p11.mean(),p12.mean()]) #povprečje x
    return xpovp

pod11 = funkcija1(10,1)
pod22 = funkcija1(10,2)
pod33 = funkcija1(10,3)
pod44 = funkcija1(10,4)
pod55 = funkcija1(10,5)
pod66 = funkcija1(10,6)
pod77 = funkcija1(10,7)
pod88 = funkcija1(10,8)
pod99 = funkcija1(10,9)
pod1010 = funkcija1(10,10)
pod1111 = funkcija1(10,11)
pod1212 = funkcija1(10,12)
pod1313 = funkcija1(10,13)

leto1011 = povprecje(pod11,pod22,pod33,pod44,pod55,pod66,pod77,pod88,pod99,pod1010,pod1111,pod1212,pod1313)

def racun(podatki):
    bins = np.arange(np.floor(podatki.min()),np.ceil(podatki.max()))
    values, base = np.histogram(podatki, bins=bins,normed=True)
    cdf = np.cumsum(values)
    a,b=base[:-1],cdf
    return a,b,values

'''primerjave med leti'''
#fig = plt.figure(1)
#sub = fig.add_subplot(131)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,2), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,2), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,2), funkcija1(13,2))
#fig.suptitle(r'Primerjava porazdelitve oddaje različnih nalog 2010/2011 in 2013/2014')
#plt.title(r'Oddaja naloge 2')
#plt.text(-24,6,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(132)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,5), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,5), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,5), funkcija1(13,5))
#
#plt.title(r'Oddaja naloge 5')
#plt.text(-24,4,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(133)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,9), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,9), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,9), funkcija1(13,9))
#plt.title(r'Oddaja naloge 9')
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#fig = plt.figure(2)
#sub = fig.add_subplot(131)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,7), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,7), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,7), funkcija1(13,7))
#fig.suptitle(r'Primerjava porazdelitve oddaje različnih nalog  med leti 2010/2011 in 2013/2014')
#plt.title(r'Oddaja naloge 7')
#plt.text(-24,4,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(132)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,11), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,11), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,11), funkcija1(13,11))
#plt.title(r'Oddaja naloge 11')
#
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(133)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(11,12), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2011/2012$')
#value2,bins2,c2=sub.hist(funkcija1(13,12), alpha=0.5, bins=np.linspace(-26, 21, 20), label='$2013/2014$')
#ks89,ksp89=stats.ks_2samp(funkcija1(11,12), funkcija1(13,12))
#plt.title(r'Oddaja naloge 12')
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 

'''primerjave mad nalogami'''
#fig = plt.figure(2)
#sub = fig.add_subplot(131)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(13,1), alpha=0.5, bins=np.linspace(-26, 21, 20), label='naloga 1')
#value2,bins2,c2=sub.hist(funkcija1(13,5), alpha=0.5, bins=np.linspace(-26, 21, 20), label='naloga 5')
#ks89,ksp89=stats.ks_2samp(funkcija1(13,1), funkcija1(13,5))
#fig.suptitle(r'Primerjava porazdelitev različnih nalog za leto 2013/14')
##plt.title(r'$Oddaje$ $naloge$ $1$')
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(132)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(13,1), alpha=0.5, bins=np.linspace(-26, 21, 20), label='naloga 1')
#value2,bins2,c2=sub.hist(funkcija1(13,7), alpha=0.5, bins=np.linspace(-26, 21, 20), label='naloga 7')
#ks89,ksp89=stats.ks_2samp(funkcija1(13,1), funkcija1(13,7))
##plt.title(r'$Oddaje$ $naloge$ $8$')
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 
#
#sub = fig.add_subplot(133)
#sub.grid()
#value1,bins1,c1=sub.hist(funkcija1(13,1),alpha=0.5,  bins=np.linspace(-26, 21, 20), label='naloga 1')
#value2,bins2,c2=sub.hist(funkcija1(13,12), alpha=0.5, bins=np.linspace(-26, 21, 20), label='naloga 12')
#ks89,ksp89=stats.ks_2samp(funkcija1(13,1), funkcija1(13,12))
##plt.title(r'$Oddaje$ $naloge$ $13$')
#plt.text(-24,2,'D=%.5f,\n p=%.5f'%(ks89,ksp89))
#plt.xlabel('t[h]')
#plt.ylabel('N')
#sub.legend(loc='best') 


'''računanje statističnih razlik'''
#chi_test89,p89= stats.chisquare(value1,  value2)
#chi_test18,p18= stats.chisquare(racun(pod88)[2],  racun(pod11)[2])
#chi_test19,p19= stats.chisquare(racun(pod11)[2],  racun(pod99)[2])
#chi_test35,p35= stats.chisquare(racun(pod33)[2],  racun(pod55)[2])
#chi_test46,p46= stats.chisquare(racun(pod44)[2],  racun(pod66)[2])
#chi_test47,p47= stats.chisquare(racun(pod44)[2],  racun(pod77)[2])
#chi_test67,p67= stats.chisquare(racun(pod77)[2],  racun(pod66)[2])
#chi_test113,p113= stats.chisquare(racun(pod11)[2],  racun(pod1313)[2])
#chi_test19,p19= stats.chisquare(racun(pod88)[2],  racun(pod11)[2])
#chi_test18,p18= stats.chisquare(racun(pod11)[2],  racun(pod99)[2])
#chi_test813,p813= stats.chisquare(racun(pod88)[2],  racun(pod1313)[2])
#chi_test913,p913= stats.chisquare(racun(pod99)[2],  racun(pod1313)[2])
#chi_test1011,p1011= stats.chisquare(racun(pod1010)[2],  racun(pod1111)[2])

'''GRAFI KUMULATIVNIH PORAZDELITEV ODDAJE NALOG ZA 10/11'''
fig=plt.figure(3)

sub= fig.add_subplot(111)
c=sub.plot(racun(pod11)[0],racun(pod11)[1], color='green', alpha=0.5, label='nal. 1')
d=sub.plot(racun(pod22)[0],racun(pod22)[1], color='lightgreen', alpha=0.5 ,label='nal. 2')
e=sub.plot(racun(pod33)[0],racun(pod33)[1], color='yellow', alpha=0.5 ,label='nal. 3')
f=sub.plot(racun(pod44)[0],racun(pod44)[1], color='orange', alpha=0.5 ,label='nal. 4')
g=sub.plot(racun(pod55)[0],racun(pod55)[1], color='red', alpha=0.5 ,label='nal. 5')
h=sub.plot(racun(pod66)[0],racun(pod66)[1], color='darkred', alpha=0.5 ,label='nal. 6')
i=sub.plot(racun(pod77)[0],racun(pod77)[1], color='violet', alpha=0.5 ,label='nal. 7')
j=sub.plot(racun(pod88)[0],racun(pod88)[1], color='darkblue', alpha=0.5 ,label='nal. 8')
k=sub.plot(racun(pod99)[0],racun(pod99)[1], color='lightblue', alpha=0.5 ,label='nal. 9')
l=sub.plot(racun(pod1010)[0],racun(pod1010)[1], color='lightgrey', alpha=0.5 ,label='nal. 10')
m=sub.plot(racun(pod1111)[0],racun(pod1111)[1], color='brown', alpha=0.5 ,label='nal. 11')
n=sub.plot(racun(pod1212)[0],racun(pod1212)[1], color='black', alpha=0.5,label='nal. 12')
o=sub.plot(racun(pod1313)[0],racun(pod1313)[1], color='aquamarine',label='nal. 12')
plt.xlim(-30,30)
plt.title(r'Primerjava kumulativne porazdelitve oddaje nalog 2010/11')
plt.xlabel('čas oddaje [h]')
plt.ylabel('F(X) = P(x < X)')
plt.legend()   
plt.grid()
fig=plt.figure(4)
sub= fig.add_subplot(111)


pod11 = funkcija1(11,1)
pod22 = funkcija1(11,2)
pod33 = funkcija1(11,3)
pod44 = funkcija1(11,4)
pod55 = funkcija1(11,5)
pod66 = funkcija1(11,6)
pod77 = funkcija1(11,7)
pod88 = funkcija1(11,8)
pod99 = funkcija1(11,9)
pod1010 = funkcija1(11,10)
pod1111 = funkcija1(11,11)
pod1212 = funkcija1(11,12)
plt.grid()
c=sub.plot(racun(pod11)[0],racun(pod11)[1], color='green', alpha=0.5, label='nal. 1')
d=sub.plot(racun(pod22)[0],racun(pod22)[1], color='lightgreen', alpha=0.5 ,label='nal. 2')
e=sub.plot(racun(pod33)[0],racun(pod33)[1], color='yellow', alpha=0.5 ,label='nal. 3')
f=sub.plot(racun(pod44)[0],racun(pod44)[1], color='orange', alpha=0.5 ,label='nal. 4')
g=sub.plot(racun(pod55)[0],racun(pod55)[1], color='red', alpha=0.5 ,label='nal. 5')
h=sub.plot(racun(pod66)[0],racun(pod66)[1], color='darkred', alpha=0.5 ,label='nal. 6')
i=sub.plot(racun(pod77)[0],racun(pod77)[1], color='violet', alpha=0.5 ,label='nal. 7')
j=sub.plot(racun(pod88)[0],racun(pod88)[1], color='darkblue', alpha=0.5 ,label='nal. 8')
k=sub.plot(racun(pod99)[0],racun(pod99)[1], color='lightblue', alpha=0.5 ,label='nal. 9')
l=sub.plot(racun(pod1010)[0],racun(pod1010)[1], color='lightgrey', alpha=0.5 ,label='nal. 10')
m=sub.plot(racun(pod1111)[0],racun(pod1111)[1], color='brown', alpha=0.5 ,label='nal. 11')
n=sub.plot(racun(pod1212)[0],racun(pod1212)[1], color='black', alpha=0.5,label='nal. 12')
o=sub.plot(racun(pod1313)[0],racun(pod1313)[1], color='aquamarine',label='nal. 12')
plt.xlim(-30,30)
plt.xlim(-30,30)
plt.title(r'Primerjava kumulativne porazdelitve oddaje nalog 2011/12')
plt.xlabel('čas oddaje [h]')
plt.ylabel('F(X) = P(x < X)')
plt.legend()   



leto1112 = povprecje1(pod11,pod22,pod33,pod44,pod55,pod66,pod77,pod88,pod99,pod1010,pod1111,pod1212)

#plt.figure(2)
#plt.grid()
#c=plt.plot(racun(pod11)[0],racun(pod11)[1], color='green', alpha=0.5)
#d=plt.plot(racun(pod22)[0],racun(pod22)[1], color='lightgreen', alpha=0.5)
#e=plt.plot(racun(pod33)[0],racun(pod33)[1], color='yellow', alpha=0.5)
#f=plt.plot(racun(pod44)[0],racun(pod44)[1], color='orange', alpha=0.5)
#g=plt.plot(racun(pod55)[0],racun(pod55)[1], color='red', alpha=0.5)
#h=plt.plot(racun(pod66)[0],racun(pod66)[1], color='darkred', alpha=0.5)
#i=plt.plot(racun(pod77)[0],racun(pod77)[1], color='violet', alpha=0.5)
#j=plt.plot(racun(pod88)[0],racun(pod88)[1], color='darkblue', alpha=0.5)
#k=plt.plot(racun(pod99)[0],racun(pod99)[1], color='lightblue', alpha=0.5)
#l=plt.plot(racun(pod1010)[0],racun(pod1010)[1], color='lightgrey', alpha=0.5)
#m=plt.plot(racun(pod1111)[0],racun(pod1111)[1], color='brown', alpha=0.5)
#n=plt.plot(racun(pod1212)[0],racun(pod1212)[1], color='black', alpha=0.5)
##plt.xlim(-40,40)
#plt.title(r'$Primerjava$ $oddaje$ $nalog$ - $2011/12$')
#plt.xlabel('$t[h]$')
#plt.ylabel('$k(z)/N$')
#plt.legend() 

pod11 = funkcija1(13,1)
pod22 = funkcija1(13,2)
pod33 = funkcija1(13,3)
pod44 = funkcija1(13,4)
pod55 = funkcija1(13,5)
pod66 = funkcija1(13,6)
pod77 = funkcija1(13,7)
pod88 = funkcija1(13,8)
pod99 = funkcija1(13,9)
pod1010 = funkcija1(13,10)
pod1111 = funkcija1(13,11)
pod1212 = funkcija1(13,12)
pod1313 = funkcija1(13,13)

leto1314 = povprecje(pod11,pod22,pod33,pod44,pod55,pod66,pod77,pod88,pod99,pod1010,pod1111,pod1212,pod1313)
fig=plt.figure(6)
plt.grid()
sub= fig.add_subplot(111)
c=sub.plot(racun(pod11)[0],racun(pod11)[1], color='green', alpha=0.5, label='nal. 1')
d=sub.plot(racun(pod22)[0],racun(pod22)[1], color='lightgreen', alpha=0.5 ,label='nal. 2')
e=sub.plot(racun(pod33)[0],racun(pod33)[1], color='yellow', alpha=0.5 ,label='nal. 3')
f=sub.plot(racun(pod44)[0],racun(pod44)[1], color='orange', alpha=0.5 ,label='nal. 4')
g=sub.plot(racun(pod55)[0],racun(pod55)[1], color='red', alpha=0.5 ,label='nal. 5')
h=sub.plot(racun(pod66)[0],racun(pod66)[1], color='darkred', alpha=0.5 ,label='nal. 6')
i=sub.plot(racun(pod77)[0],racun(pod77)[1], color='violet', alpha=0.5 ,label='nal. 7')
j=sub.plot(racun(pod88)[0],racun(pod88)[1], color='darkblue', alpha=0.5 ,label='nal. 8')
k=sub.plot(racun(pod99)[0],racun(pod99)[1], color='lightblue', alpha=0.5 ,label='nal. 9')
l=sub.plot(racun(pod1010)[0],racun(pod1010)[1], color='lightgrey', alpha=0.5 ,label='nal. 10')
m=sub.plot(racun(pod1111)[0],racun(pod1111)[1], color='brown', alpha=0.5 ,label='nal. 11')
n=sub.plot(racun(pod1212)[0],racun(pod1212)[1], color='black', alpha=0.5,label='nal. 12')
o=sub.plot(racun(pod1313)[0],racun(pod1313)[1], color='aquamarine',label='nal. 12')
plt.xlim(-30,30)
plt.xlim(-30,30)
plt.title(r'Primerjava kumulativne porazdelitve oddaje nalog 2013/14')
plt.xlabel('čas oddaje [h]')
plt.ylabel('F(X) = P(x < X)')
plt.legend()   
plt.grid()

pod11 = funkcija1(14,1)
pod22 = funkcija1(14,2)
pod33 = funkcija1(14,3)
pod44 = funkcija1(14,4)
pod55 = funkcija1(14,5)
pod66 = funkcija1(14,6)
pod77 = funkcija1(14,7)
pod88 = funkcija1(14,8)
pod99 = funkcija1(14,9)
pod1010 = funkcija1(14,10)
pod1111 = funkcija1(14,11)
pod1212 = funkcija1(14,12)
pod1313 = funkcija1(14,13)

leto1415 = povprecje(pod11,pod22,pod33,pod44,pod55,pod66,pod77,pod88,pod99,pod1010,pod1111,pod1212,pod1313)

fig=plt.figure(7)
plt.grid()
sub= fig.add_subplot(111)
c=plt.plot(racun(pod11)[0],racun(pod11)[1], color='green', alpha=0.5)
d=plt.plot(racun(pod22)[0],racun(pod22)[1], color='lightgreen', alpha=0.5)
e=plt.plot(racun(pod33)[0],racun(pod33)[1], color='yellow', alpha=0.5)
f=plt.plot(racun(pod44)[0],racun(pod44)[1], color='orange', alpha=0.5)
g=plt.plot(racun(pod55)[0],racun(pod55)[1], color='red', alpha=0.5)
h=plt.plot(racun(pod66)[0],racun(pod66)[1], color='darkred', alpha=0.5)
i=plt.plot(racun(pod77)[0],racun(pod77)[1], color='violet', alpha=0.5)
j=plt.plot(racun(pod88)[0],racun(pod88)[1], color='darkblue', alpha=0.5)
k=plt.plot(racun(pod99)[0],racun(pod99)[1], color='lightblue', alpha=0.5)
l=plt.plot(racun(pod1010)[0],racun(pod1010)[1], color='lightgrey', alpha=0.5)
m=plt.plot(racun(pod1111)[0],racun(pod1111)[1], color='brown', alpha=0.5)
n=plt.plot(racun(pod1212)[0],racun(pod1212)[1], color='black', alpha=0.5)
o=plt.plot(racun(pod1313)[0],racun(pod1313)[1], color='aquamarine')
plt.xlim(-40,40)
plt.title(r'Primerjava kumulativne porazdelitve oddaje nalog 2011/12')
plt.xlabel('čas oddaje [h]')
plt.ylabel('F(X) = P(x < X)')
plt.legend() 

plt.figure(8)
plt.grid()
l1011=plt.plot(racun(leto1011)[0],racun(leto1011)[1], color='green', alpha=0.8, label='$2010/2011$')
l1112=plt.plot(racun(leto1112)[0],racun(leto1112)[1], color='lightgreen', alpha=0.8, label='$2011/2012$')
l1314=plt.plot(racun(leto1314)[0],racun(leto1314)[1], color='lightblue', alpha=0.8,label='$2013/2014$')
l1415=plt.plot(racun(leto1415)[0],racun(leto1415)[1], color='blue', alpha=0.8, label='$2014/2015$')
plt.title(r'$Primerjava$ $povprečne$ $oddaje$ $nalog$')
plt.xlabel('$t[h]$')
plt.ylabel('$k(z)/N$')
plt.legend() 
#












bins = np.arange(np.floor(pod22.min()),np.ceil(pod22.max()))
values, base = np.histogram(pod22, bins=bins, density=1,normed=True)
cdf = np.cumsum(values)
b=plt.plot(base[:-1], cdf,color='lightgreen')
bins = np.arange(np.floor(pod33.min()),np.ceil(pod33.max()))
values, base = np.histogram(pod33, bins=bins, density=1,normed=True)
cdf = np.cumsum(values)
c=plt.plot(base[:-1], cdf,)
bins = np.arange(np.floor(pod44.min()),np.ceil(pod44.max()))
values, base = np.histogram(pod44, bins=bins, density=1,normed=True)
cdf = np.cumsum(values)
d=plt.plot(base[:-1], cdf,color='orange')

