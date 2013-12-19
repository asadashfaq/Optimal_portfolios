#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

Optimal_values = np.loadtxt('Optimal_values02') # Loading the standard deviation and capacity factor
Quantiles = np.loadtxt('Optimal_quantiles')
Var1 = np.loadtxt('var1')
Var2 = np.loadtxt('var2')
Var3 = np.loadtxt('var3')
Var6 = np.loadtxt('var6')
Sol_var1 = np.loadtxt('sol_var1')
Sol_var2 = np.loadtxt('sol_var2')
Sol_var3 = np.loadtxt('sol_var3')
Sol_var6 = np.loadtxt('sol_var6')
#Capacity = np.loadtxt('Capacity_values02') # Loading the installed capacity
Optimal_values2 = np.loadtxt('Solar_Optimal_values')
#Gamma_values = np.loadtxt('Gamma_values')
#b = np.loadtxt('b')
#Indices = [int(b[x]) for x in range(len(b))]
Std=list(Optimal_values[::2]) # picking the standard deviations from Optimal_values
Cf=list(Optimal_values[1::2]) # picking the capacity factor from Optimal_values
#Cap=list(Capacity)
Std2=list(Optimal_values2[::2])
Cf2=list(Optimal_values2[1::2])
#Gamma = [Gamma_values[x*30:(x+1)*30] for x in range(len(Gamma_values)/30)]

#Good_gamma =  [Gamma[Indices[x]] for x in range(len(Indices))]
#c = [Good_gamma[x][y] for x in range(len(Indices)) for y in range(30)]
#f = open('Good_gamma', "a")
#for x in range(len(c)):
#	f.write(str(c[x]) + '\n')
#f.close()
plt.ion()
#plt.subplot(211)
plt.plot(Std,Cf,'bo')
#plt.xlabel('Standard deviation (GW)')
#plt.ylabel('Capacity factor')
#plt.title('Capacity factor vs the standard deviation in EU for gamma values between 0.8 and 1.2')
#plt.axis([140, 210, 0.22, 0.36])
#plt.plot(gamma_1_std,gamma_1_cf,'ro')
#plt.subplot(212)
plt.plot(Std2,Cf2,'bo')
#plt.plot(Var1,Cf,'bo')
#plt.plot(Quantiles,Cf,'bo')
#plt.plot(Quantiles,Cf,'bo')
plt.annotate('Wind power',(210,0.27),size=20)
plt.annotate('Solar power',(370,0.19),size=20)
plt.xlabel('Standard deviation (GW)',size=20)
plt.ylabel('Capacity factor',size=20)
plt.tick_params(labelsize='larger')
#plt.annotate('Wind Power',(200,0.27),size=20)
#plt.annotate('Solar Power',(450,0.22),size=20)

"""
var = []
a = np.loadtxt('a')
for x in range(len(a)):
	var.append(int(a[x]))
	plt.plot(Sol_var6[var[x]],Cf2[var[x]],'ro',markersize=10)
"""	

plt.show()
