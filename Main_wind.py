#!/usr/bin/python
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show

######################################################################################
############################## Loading data ##########################################
######################################################################################

Optimal_values = np.loadtxt('Data_files/Optimal_values02') #Load the standard deviation and capacity factor
Gamma_values = np.loadtxt('Data_files/Gamma_values02') #Load the gamma-values
Std = (Optimal_values[::2]) #Pick out the standard deviations from Optimal_values
Cf = (Optimal_values[1::2]) #Pick out the capacity factor from Optimal_values
Gamma=np.array([Gamma_values[30*x:(x+1)*30] for x in range(len(Std))]) #Converting the gamma-values in listform into 30-element arrays.
cf = list(Cf) #Make capacity factor into a list, so we can use index later (cause I have no idea how to do it with arrays ;) )
#Optimal_quantiles = np.loadtxt('Optimal_quantiles')
#Optimal_quantiles_index = np.loadtxt('b')

######################################################################################
############## Calculating key properties for Wind, Solar and Load ###################
######################################################################################

# Files[24:26] is AA
Files = list('Data_files/ISET_country_AA.npz')
Country_names = ['AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK']
# This for-loop is used to replace AA with the country codes, and then combine the list into a string
for x in range(len(Country_names)):
	Files[24:26] = Country_names[x]
	Filename = ''.join(Files)
	Country_names[x] = np.load(Filename)

# We load the wind-data from the ISET-files and calculate the standard deviation of the individual countries.
Wind = [Country_names[x]['Gw'] for x in range(len(Country_names))]
Wind_std = [np.std(Wind[x]) for x in range(len(Country_names))]
corr = np.corrcoef(Wind) #corrcoef is the correlation between elements in an array
EU_W = sum(Wind)
Wind_cf = [1/max(Wind[x]) for x in range(len(Wind))] #Wind_cf is just the capacity factor of the wind

Solar = [Country_names[x]['Gs'] for x in range(len(Country_names))]
Solar_std = [np.std(Solar[x]) for x in range(len(Country_names))]
corr_solar = np.corrcoef(Solar)
EU_S = sum(Solar)
Solar_cf = [1/max(Solar[x]) for x in range(len(Wind))]

# We load the load-data from the ISET-files and calculate the mean load
Load = [Country_names[x]['L'] for x in range(len(Country_names))]
Mean_load = np.array([np.mean(Load[j]) for j in range(30)])
initial_total_demand = sum([np.mean(Load[x]) for x in range(30)])
EU_L =sum(Load)

######################################################################################
##### Making gamma; calculating Standard deviation and Capacity factor ###############
######################################################################################

#This function makes random numbers, and then add a correction, beta, to match the constraint that gamma_eu = 1
def make_gamma():
	gamma_0 = np.random.random(30)*2
	beta = sum(gamma_0*Mean_load)/EU_L.mean() - 1
	gamma = gamma_0/(1+beta)
	return gamma

#This function makes the reference point where gamma = 1 in each country
def make_gamma_1():
	gamma_1 = np.ones(30)
	return gamma_1

#Calculating the total standard deviation
def total_std(gammas):
	A = 0; B = 0
	for x in range(30):
		A += Mean_load[x]**2*gammas[x]**2*Wind_std[x]**2
		for y in range(30):
			if y != x:
				B += (gammas[x]*gammas[y]*Mean_load[x]*Mean_load[y]*corr[x,y]*Wind_std[x]*Wind_std[y])
	return np.sqrt(A+B)


#Calculating the total capacity factor
def total_cf(gammas):
	wind_capacity = gammas*Mean_load/Wind_cf
	total_wind_capacity = sum(wind_capacity)
	Total_cf = EU_L.mean()/total_wind_capacity
	return Total_cf

#Calculating the total installed capacity
def wind_capacity(gammas):
	wind_capacity = gammas*Mean_load/Wind_cf
	total_wind_capacity = sum(wind_capacity)
	return total_wind_capacity

######################################################################################
################## Finding optimal gamma values ######################################
######################################################################################

#e is the optimal gammas found in each interval. new_std and new_cf are the capacity factor and standard deviation corresponding to those optimal gammas
e = []; new_std = []; new_cf = []
def find_optimal_gamma():
	for x in range(4):
		over= np.array(list((Std>147+8*x).nonzero())) #Picks standard deviation values above 147 + 4*x
		under = np.array(list((Std<155+8*x).nonzero())) #Picks standard deviation values below 151 + 4*x
		indices = np.intersect1d(over,under) #Finding the indices of the values that are both in "over" and "under"
		list_cf = list(Cf[indices]) #making the capacity factors in the interval into a list
		top_values = ([cf.index(hq.nlargest(100,list_cf)[y]) for y in range(100)]) #Finds the top n values in the interval
		average_gamma = (np.average(Gamma[top_values], axis=0))*Mean_load/Wind_cf
		e.append(average_gamma)
		new_std.append(total_std(average_gamma))
		new_cf.append(total_cf(average_gamma))
	return e

a = total_std(make_gamma_1())
b = total_cf(make_gamma_1())
e = find_optimal_gamma()
f = []
for x in range(30):
	for y in range(4):
		f.append(e[y][x])
e.append(make_gamma_1()) #adds the reference point

######################################################################################
####################### Calculating Mismatch and Balancing energy ####################
######################################################################################

#Calculates the mismatch energy for each time-step and returns 70128*x (where x is the number of intervals of alpha we're using) element list. (DO NOT TRY TO OPEN THIS!!)
def mismatch_energy(gammas):
	Wind_power = 0
	for x in range(30):
		Wind_power += (Wind[x]*gammas[x]*Mean_load[x])
	Solar_power = 0
	for x in range(30):
		Solar_power += Solar[x]*Mean_load[x]
	Load_power = 0	
	for x in range(30):
		Load_power += Load[x]
	Delta_EU = []
	for x in range(10):
		Delta_EU.append(((x/10.0)*Wind_power + (1.0-x/10.0)*Solar_power) - Load_power)	
	return Delta_EU	

#Calculates the balancing energy by setting all positive values of the mismatch energy equal to 0, and then calculate the average.
def balancing_energy(gammas):
	mismatch = mismatch_energy(gammas)
	negative_mismatch_index = [mismatch[y] < 0 for y in range(len(mismatch))]
	negative_mismatch = [negative_mismatch_index[x]*mismatch[x] for x in range(len(mismatch))]
	total_mismatch_energy = []
	for x in range(10):
		average = np.average(negative_mismatch[x])*(-1)
		total_mismatch_energy.append(np.sum(average))
	return total_mismatch_energy

######################################################################################
############################# Calculating 1 % quantiles ##############################
######################################################################################

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

def integration(gammas):
	b = mismatch_energy(gammas)
	#i = []; h = []	
	#c = min(b[x])
	#d = max(b[x])
	#e = np.percentile(b[x],99)
	f = np.percentile(b,1)
	#g = list(np.sort(b[x]))
	#h.append(find_nearest(b[x],e))
	i = (find_nearest(b,f))
	#j = len(g[0:g.index(i)])
	#k = len(g[g.index(h):-1])
	return i

######################################################################################
################ Calculating variability, plotting variance of variability ###########
######################################################################################

#Calculates the variability, the difference in mismatches at certain time intervals. Returns dDelta, the variability and std, the variance.
def variability(gammas):
	mismatch = mismatch_energy(gammas)
	dDelta = []; variance = []
	for z in range(10):
		for x in [1,2,3,6]:
			for y in range(len(mismatch[0])):
				dDelta.append(mismatch[z][y]-mismatch[z][y-x])
	for x in range(40):
		variance.append(np.var(dDelta[70128*x:70128*(x+1)]))
	std = np.array([variance[x*6:(x+1)*6] for x in range(10)])
	return dDelta, std

"""
#Plotting the variance of the variability for different time intervals
titles = ['layout 1','layout 2','layout 3','layout 4','ref.point']
shapes = ['ro','g<','bs','ys','ro','r>']
for x in range(5):
	a, b = variability(e[x])
	steps = 231+x
	plt.subplot(steps)	
	plt.semilogy() #make sure the y-axis is logarithmic		
	for y in range(6):
		plt.plot(b,shapes[y])
	plt.legend(('24h','50h','100h','200h','300h','400h'))	
	plt.xlabel('alpha')
	plt.ylabel('variance (GW^2)')
	plt.title(titles[x])
plt.show()
"""

"""
#Plotting the variability histogram in 5 different figures (one for each layout) with the different time intervals. 
plt.ion()
titles = ['1h','3h','7h','24h','72h','144h']
titles2 = ['layout 1','layout 2','layout 3','layout 4','ref.point']
for y in range(5):
	plt.figure(y)
	for x in range(6):
		steps = 231+x
		plt.subplot(steps)
		#plt.axis([-400,400,0,30000])
		plt.title(titles[x])
		plt.xlabel('Mismatch power (GW)')
		a, b = variability(e[y])
		plt.hist(a[(x*70128+0*70128):((x+1)*70128+0*70128)],bins=100) #to use alpha = 0.8
	plt.suptitle(titles2[y] + ' alpha = 0.0', fontsize=16)
	plt.show()
"""
######################################################################################
############# Finding 1 % quantiles in mismatch distribution##########################
######################################################################################
"""
titles = ['Layout 1','Layout 2','Layout 3','Layout 4','Ref.point']
for x in range(5):
	a = mismatch_energy(e[x])[0]
	steps = 231+x
	plt.subplot(steps)
	plt.title(titles[x],size=20)
	low_quan = integration(e[x])[0]
	high_quan = integration(e[x])[1]
	minpoint = a[low_quan]
	maxpoint = a[high_quan]
	histplot = plt.hist(a,bins=100,alpha=0.2)
	histploty = histplot[0]
	histplotx = histplot[1]
	low_height = histploty[find_nearest(histplotx,minpoint)]
	high_height = histploty[find_nearest(histplotx,maxpoint)]
	plt.vlines(minpoint,0,ymax=low_height)
	plt.vlines(maxpoint,0,ymax=high_height)
	plt.vlines(min(a),0,ymax=low_height)
	plt.xlabel('Mismatch (GW)',size=20)
	plt.tick_params(labelsize='large')
	mina = round(min(a),2)
	diff = round(minpoint-min(a),2)
	quan = round(minpoint,2)
	plt.annotate('minimum \n' + str(mina),(mina-14,low_height+40))
	plt.annotate('interval \n' + str(diff),((mina+minpoint)/2-18,low_height-50))
	plt.annotate('1 % quantile \n' + str(quan),(minpoint-19,low_height+40))
	plt.axis([-400,-240,0,500])
"""
######################################################################################
##################### Calculate and plot 1 % quantiles ###############################
######################################################################################
"""
shapes = ['r<','r>','rs','ro']
d = integration(make_gamma_1())
plt.plot(Optimal_quantiles[d],b,'yo',markersize=12)
for x in range(4):
	c = Optimal_quantiles_index[x]
	plt.plot(Optimal_quantiles[c],Cf[c],shapes[x],markersize=12)
	plt.legend(('ref.point','layout 1','layout 2','layout 3','layout 4','layout 5'))
plt.plot(Optimal_quantiles,Cf,'bo')
for x in range(4):
	c = Optimal_quantiles_index[x]
	plt.plot(Optimal_quantiles[c],Cf[c],shapes[x],markersize=12)
d = integration(make_gamma_1())
plt.plot(Optimal_quantiles[d],b,'yo',markersize=12)
plt.xlabel('1 % quantiles (GW)',size=20)
plt.ylabel('Capacity factor',size=20)
plt.tick_params(labelsize='larger')
#plt.suptitle('4 Optimal layouts for Solar only',size=20)
plt.show()
"""
######################################################################################
##################### Plotting balancing energy and comparing gammas #################
######################################################################################

"""
#Plotting the the average balancing energy as a function of alpha.
shapes = ['ro','g<','b>','ms','c-']
plt.ion()
for x in range(5):
	balancingenergy = balancing_energy(e[x])
	xcoord = np.linspace(0.01,1.00,10)
	plt.plot(xcoord,balancingenergy,shapes[x])
	plt.xlabel('alpha',size=20)
	plt.ylabel('absolute value of balancing energy',size=20)
	plt.legend(('layout 1','layout 2', 'layout 3', 'layout 4','ref.point'))
	plt.tick_params(labelsize='larger')
	#plt.suptitle('Balancing energy in a model with heterogenous wind and homogeneous solar power',size=20)

"""

"""
my_colors = 'rrrrbbbb' #A very brilliant way of coloring every 10 bars red and blue
plt.bar(range(len(f)), f, width=0.5, bottom=0, color=my_colors)
ind = np.arange(30)*4 + 2
plt.tick_params(labelsize='large')
plt.xticks(ind, ('AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK'))
#plt.suptitle('Gamma values for 4 different optimal wind layouts for each country in EU(30)',size=20)
plt.ylabel('Installed wind capacity in each country (GW)',size=20)
plt.show()
"""

######################################################################################
#################### Saving standard deviation, capacity factor and gamma values #####
######################################################################################

"""
f = open('Data_files/Optimal_values03', "a") #Makes (or add to the existing) 'Optimal_values02'
h = open('Data_files/Gamma_values03', "a") #Makes (or add to the existing) 'Gamma_values02'
i = open('Data_files/Capacity_values03', "a") #Makes (or add to the existing) 'Capacity_values02'

std_value = []; cf_value = []; gamma_value = []; capacity = []
plt.ion()
for x in range(1000): #Number of points we want to make
	g = make_gamma() # calling the gamma-making function
	#w = generate_wind(g)
	std = total_std(g) #calculate standard deviation
	cf = total_cf(g) #calculate capacity factor
	cap = wind_capacity(g) #calculate installed wind capacity
	std_value.append(std); cf_value.append(cf); gamma_value.append(g); capacity.append(cap) #making standard deviation, capacity factor, gamma-values and installed capacities into lists.
	plt.ion()
	plt.subplot(211)
	plt.plot(std,cf,'bo')
	plt.xlabel('Standard deviation (GW)')
	plt.ylabel('Capacity factor')
	plt.title('Capacity factor vs the standard deviation in EU for gamma values between 0.8 and 1.2')
	plt.subplot(212)
	plt.plot(std,cap,'bo')
	plt.xlabel('Standard deviation (GW)')
	plt.ylabel('Installed capacity (GW)')
	plt.title('Installed capacity vs the standard deviation in EU for gamma-values between 0.8 and 1.2')
	f.write(str(std_value[x]) + '\n' + str(cf_value[x]) + '\n')
	for y in range(30):
		h.write(str(gamma_value[x][y]) + '\n')
	i.write(str(capacity[x]) + '\n')

#finding the standard deviation, capacity factor and installed capacity in the case of gamma = 1 for each country.
gamma_1 = np.ones(30)
gamma_1_std = total_std(gamma_1)
gamma_1_cf = total_cf(gamma_1)
gamma_1_std = total_std(gamma_1)
gamma_1_c = wind_capacity(gamma_1)
plt.subplot(211)
plt.plot(gamma_1_std,gamma_1_cf,'ro')
plt.subplot(212)
plt.plot(gamma_1_std,gamma_1_c,'ro')
plt.show()
i.close()
h.close()
f.close()
"""
