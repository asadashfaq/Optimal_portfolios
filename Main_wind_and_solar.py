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
Gamma_wind=np.array([Gamma_values[30*x:(x+1)*30] for x in range(len(Std))]) #Converting the gamma-values in listform into 30-element arrays.
cf = list(Cf) #Make capacity factor into a list, so we can use index later (cause I have no idea how to do it with arrays ;) )
Optimal_values_solar = np.loadtxt('Data_files/Solar_Optimal_values') #Load the standard deviation and capacity factor
Gamma_values_solar = np.loadtxt('Data_files/Solar_Gamma_values') #Load the gamma-values
Std_solar = (Optimal_values_solar[::2]) #Pick out the standard deviations from Optimal_values
Cf_solar = (Optimal_values_solar[1::2]) #Pick out the capacity factor from Optimal_values
Gamma_solar=np.array([Gamma_values_solar[30*x:(x+1)*30] for x in range(len(Std_solar))]) #Converting the gamma-values in listform into 30-element arrays.
cf_solar = list(Cf_solar) #Make capacity factor into a list, so we can use index later (cause I have no idea how to do it with arrays ;) )
Optimal_wind_gamma_index = np.loadtxt('Data_files/Wind_optimal_gammas')
Optimal_solar_gamma_index = np.loadtxt('Data_files/Solar_optimal_gammas')

# Files[24:26] is AA
Files = list('Data_files/ISET_country_AA.npz')
Country_names = ['AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK']
# This for-loop is used to replace AA with the country codes, and then combine the list into a string
for x in range(len(Country_names)):
	Files[24:26] = Country_names[x]
	Filename = ''.join(Files)
	Country_names[x] = np.load(Filename)

######################################################################################
############## Calculating key properties for Wind, Solar and Load ###################
######################################################################################

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

#Calculating the total standard deviation for wind
def total_std_wind(gammas):
	A = 0; B = 0
	for x in range(30):
		A += Mean_load[x]**2*gammas[x]**2*Wind_std[x]**2
		for y in range(30):
			if y != x:
				B += (gammas[x]*gammas[y]*Mean_load[x]*Mean_load[y]*corr[x,y]*Wind_std[x]*Wind_std[y])
	return np.sqrt(A+B)

#Calculating the total standard deviation for solar
def total_std_solar(gammas):
	A = 0; B = 0
	for x in range(30):
		A += Mean_load[x]**2*gammas[x]**2*Solar_std[x]**2
		for y in range(30):
			if y != x:
				B += (gammas[x]*gammas[y]*Mean_load[x]*Mean_load[y]*corr_solar[x,y]*Solar_std[x]*Solar_std[y])
	return np.sqrt(A+B)

#Calculating the total capacity factor for wind
def total_cf_wind(gammas):
	wind_capacity = gammas*Mean_load/Wind_cf
	total_wind_capacity = sum(wind_capacity)
	Total_cf = EU_L.mean()/total_wind_capacity
	return Total_cf

#Calculating the total capacity factor for solar
def total_cf_solar(gammas):
	solar_capacity = gammas*Mean_load/Solar_cf
	total_solar_capacity = sum(solar_capacity)
	Total_cf = EU_L.mean()/total_solar_capacity
	return Total_cf


"""
#Calculating the total installed capacity
def wind_caacity(gammas):
	wind_capacity = gammas*Mean_load/Wind_cf
	total_wind_capacity = sum(wind_capacity)
	return total_wind_capacity
"""
######################################################################################
####################### Calculating Mismatch and Balancing energy ####################
######################################################################################

#Calculates the mismatch energy for each time-step and returns 70128*x (where x is the number of intervals of alpha we're using) element list. (DO NOT TRY TO OPEN THIS!!)
def mismatch_energy(gamma_wind,gamma_solar):
	Wind_power = 0
	for x in range(30):
		Wind_power += (Wind[x]*gamma_wind[x]*Mean_load[x])
	Solar_power = 0
	for x in range(30):
		Solar_power += Solar[x]*Mean_load[x]*gamma_solar[x]
	Load_power = 0	
	for x in range(30):
		Load_power += Load[x]
	Delta_EU = []
	for x in [0.0,0.6,0.8,1.0]:
		Delta_EU.append(((x/1.0)*Wind_power + (1.00-x/1.0)*Solar_power) - Load_power)	
	return Delta_EU	


#Calculates the balancing energy by setting all positive values of the mismatch energy equal to 0, and then calculate the average.
def balancing_energy(gamma_wind,gamma_solar):
	mismatch = mismatch_energy(gamma_wind,gamma_solar)
	negative_mismatch_index = [mismatch[y] < 0 for y in range(len(mismatch))]
	negative_mismatch = [negative_mismatch_index[x]*mismatch[x] for x in range(len(mismatch))]
	total_mismatch_energy = []
	for x in range(10):
		average = np.average(negative_mismatch[x])*(-1)
		total_mismatch_energy.append(average)
	return total_mismatch_energy
	
######################################################################################
############################# Calculating 1 % quantiles ##############################
######################################################################################

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

def integration(gamma_wind,gamma_solar):
	b = mismatch_energy(gamma_wind,gamma_solar)
	i = []; h = []	
	#c = min(b[x])
	#d = max(b[x])
	for x in range(len(b)):
		e = np.percentile(b[x],99)
		f = np.percentile(b[x],1)
		#g = list(np.sort(b[x]))
		h.append(find_nearest(b[x],e))
		i.append(find_nearest(b[x],f))
	#j = len(g[0:g.index(i)])
	#k = len(g[g.index(h):-1])
	return i, h

######################################################################################
################## Finding optimal gamma values ######################################
######################################################################################
"""
#e is the optimal gammas found in each interval. new_std and new_cf are the capacity factor and standard deviation corresponding to those optimal gammas
e = []; new_std = []; new_cf = []
def find_optimal_gamma():
	for x in range(4):
		over= np.array(list((Std>147+8*x).nonzero())) #Picks standard deviation values above 147 + 4*x
		under = np.array(list((Std<155+8*x).nonzero())) #Picks standard deviation values below 151 + 4*x
		indices = np.intersect1d(over,under) #Finding the indices of the values that are both in "over" and "under"
		list_cf = list(Cf[indices]) #making the capacity factors in the interval into a list
		top_values = ([cf.index(hq.nlargest(100,list_cf)[y]) for y in range(100)]) #Finds the top n values in the interval
		average_gamma = (np.average(Gamma[top_values], axis=0))#*Mean_load/Wind_cf
		e.append(average_gamma)
		new_std.append(total_std(average_gamma))
		new_cf.append(total_cf(average_gamma))
	return e

e = find_optimal_gamma()
f = []
for x in range(30):
	for y in range(4):
		f.append(e[y][x])
e.append(make_gamma_1()) #adds the reference point
"""

######################################################################################
################ Calculating variability, plotting variance of variability ###########
######################################################################################


#Calculates the variability, the difference in mismatches at certain time intervals. Returns dDelta, the variability and std, the variance. dDelta is a 70128*16 entrance list, so please don't open it.
def variability(gamma_wind,gamma_solar):
	mismatch = mismatch_energy(gamma_wind,gamma_solar)
	#positive_mismatch = [mismatch[0][x] > 0 for x in range(len(mismatch[0]))]*mismatch[0]
	for y in range(4):
		negative_mismatch = [(mismatch[y][x] < 0)*mismatch[y] for x in range(len(mismatch[y])) for y in range(4)]
	dDelta = []; variance = []; positive_var = []; negative_var = [];
	#for x in [1,2,3,6]:
	#	for y in range(len(mismatch[0])):
	#		positive_var.append(positive_mismatch[y]-positive_mismatch[y-x])
	for x in [1,2,3,6]:
		for y in range(len(mismatch[0])):
			negative_var.append(negative_mismatch[y]-negative_mismatch[y-x])
	#for x in range(4):
	#	variance.append(np.var(dDelta[70128*x:70128*(x+1)]))
	#std = np.array([variance[x*4:(x+1)*4] for x in range(4)])
	return negative_var

for x in range(4):
	a = variability(Gamma_wind[Optimal_wind_gamma_index[0]],Gamma_solar[Optimal_solar_gamma_index[0]])
	
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
plt.suptitle('Alpha = 1.0',size=30)
#titles = ['Optimal wind layout 1 vs optimal solar layout','Optimal wind layout 2 vs optimal solar layout','Optimal wind layout 3 vs optimal solar layout','Optimal wind layout 4 vs optimal solar layout']
titles = ['Optimal wind layout 1 vs solar reference layout','Optimal wind layout 2 vs solar reference layout','Optimal wind layout 3 vs solar reference layout','Optimal wind layout 4 vs solar reference layout']
for x in range(4):
	a = mismatch_energy(Gamma_wind[Optimal_wind_gamma_index[x]],make_gamma_1())[2]
	steps = 221+x
	plt.subplot(steps)
	plt.title(titles[x],size=20)
	low_quan, high_quan = integration(Gamma_wind[Optimal_wind_gamma_index[x]],make_gamma_1())
	#high_quan = integration(Gamma_wind[Optimal_wind_gamma_index[x]],Gamma_solar[Optimal_solar_gamma_index[0]])[1]
	minpoint = a[low_quan[2]]
	maxpoint = a[high_quan[2]]
	histplot = plt.hist(a,bins=100,alpha=0.2)
	histploty = histplot[0]
	histplotx = histplot[1]
	low_height = histploty[find_nearest(histplotx,minpoint)]
	high_height = histploty[find_nearest(histplotx,maxpoint)]
	plt.vlines(minpoint,0,ymax=low_height)
	plt.vlines(maxpoint,0,ymax=high_height)
	plt.vlines(min(a),0,ymax=low_height)
	plt.vlines(max(a),0,ymax=high_height)
	plt.xlabel('Mismatch (GW)',size=20)
	plt.tick_params(labelsize='large')
	maxa = round(max(a),2)
	diffmax = round(max(a)-maxpoint,2)
	quan = round(maxpoint,2)
	mina = round(min(a),2)
	diffmin = round(minpoint-min(a),2)
	quan = round(minpoint,2)
	plt.annotate('minimum \n' + str(mina),(mina-14,low_height+40))
	plt.annotate('interval \n' + str(diffmin),((mina+minpoint)/2-18,low_height-50))
	plt.annotate('1 % quantile \n' + str(quan),(minpoint-19,low_height+40))
	plt.annotate('maximum \n' + str(maxa),(maxa-14,high_height+40))
	plt.annotate('interval \n' + str(diffmax),((maxa+maxpoint)/2-18,high_height-50))
	plt.annotate('99 % quantile \n' + str(quan),(maxpoint-19,high_height+40))
	plt.axis([-500,600,0,2500])
"""
######################################################################################
##################### Plotting balancing energy and comparing gammas #################
######################################################################################
"""
#Plotting the the average balancing energy as a function of alpha.
shapes1 = ['ro','go','bo','mo']
shapes2 = ['r-','g-','b-','m-']
shapes3 = ['k-']
shapes4 = ['y-']
plt.ion()
xcoord = np.linspace(0.01,1.00,10)
for x in range(4):
	balancingenergy = balancing_energy(Gamma_wind[Optimal_wind_gamma_index[x]],Gamma_solar[Optimal_solar_gamma_index[0]])
	plt.plot(xcoord,balancingenergy,shapes1[x])
	
for x in range(4):
	balancingenergy2 = balancing_energy(Gamma_wind[Optimal_wind_gamma_index[x]],make_gamma_1())
	plt.plot(xcoord,balancingenergy2,shapes2[x])
balancingenergy3 = balancing_energy(make_gamma_1(),Gamma_solar[Optimal_solar_gamma_index[0]])	
plt.plot(xcoord,balancingenergy3,shapes3[0])
balancingenergy4 = balancing_energy(make_gamma_1(),make_gamma_1())
plt.plot(xcoord,balancingenergy4,shapes4[0])

plt.xlabel('alpha')	
plt.ylabel('absolute value of balancing energy')
plt.legend(('wind layout 1 vs optimal solar','wind layout 2 vs optimal solar','wind layout 3 vs optimal solar','wind layout 4 vs optimal solar','wind layout 1 vs reference solar','wind layout 2 vs reference solar','wind layout 3 vs reference solar','wind layout 4 vs reference solar','reference solar vs optimal solar','reference wind vs reference solar'))
plt.title('Balancing energy for various combinations of solar and wind layouts', size=20)
plt.show()
"""

"""
my_colors = 'rrrrbbbb' #A very brilliant way of coloring every 4 bars red and blue
plt.bar(range(len(f)), f, width=0.5, bottom=0, color=my_colors)
ind = np.arange(30)*4 + 2
plt.xticks(ind, ('AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK'))
plt.suptitle('Gamma values for 4 different optimal wind layouts for each country in EU(30)',size=20)
plt.ylabel('Gamma of individual countries')
plt.show()
"""

######################################################################################
##################### Comparing the optimal layouts of solar and wind ################
######################################################################################
"""
a = mismatch_energy(make_gamma_1(),make_gamma_1())
plt.hist(a[1:3],bins=100)
plt.legend(['Alpha = 0.6','Alpha = 0.8'])
plt.title('Wind Reference Layout vs Solar Reference Layout',size=20)
plt.axis([-500, 700, 0, 2500])
plt.xlabel('Mismatch Energy (GW)')
plt.ylabel('Hours')
"""
