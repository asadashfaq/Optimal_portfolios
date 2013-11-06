#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# Files[13:15] is AA
Files = list('ISET_country_AA.npz')
Country_names = ['AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK']
# This for-loop is used to replace AA with the country codes, and then combine the list into a string
for x in range(len(Country_names)):
	Files[13:15] = Country_names[x]
	Filename = ''.join(Files)
	Country_names[x] = np.load(Filename)

# We load the wind-data from the ISET-files and calculate the standard deviation of the individual countries.
Wind = [Country_names[x]['Gw'] for x in range(len(Country_names))]
Wind_std = [np.std(Wind[x]) for x in range(len(Country_names))]
corr = np.corrcoef(Wind) #corrcoef is the correlation between elements in an array
EU_W = sum(Wind)
Wind_cf = [1/max(Wind[x]) for x in range(len(Wind))]

# We load the load-data from the ISET-files and calculate the mean load
Load = [Country_names[x]['L'] for x in range(len(Country_names))]
Mean_load = np.array([np.mean(Load[j]) for j in range(30)])
initial_total_demand = sum([np.mean(Load[x]) for x in range(30)])
EU_L =sum(Load)

#This function makes random numbers, and then add a correction, beta, to match the constraint that gamma_eu = 1
def make_gamma():
	gamma_0 = np.random.random(30)*2
	beta = sum(gamma_0*Mean_load)/EU_L.mean() - 1
	gamma = gamma_0/(1+beta)
	return gamma


#def generate_wind(gammas):
#	wind = np.zeros(70128)
#	for i in range(30):
#		wind+=gammas[i] * Wind[i] * Mean_load[i]
#	return wind

#Calculating the total standard deviation
def total_std(gammas):
	A = 0; B = 0
	for x in range(30):
		A += np.sum(Mean_load[x]**2*gammas[x]**2*Wind_std[x]**2)
		for y in range(30):
			if y != x:
				B += np.sum(gammas[x]*gammas[y]*Mean_load[x]*Mean_load[y]*corr[x,y]*Wind_std[x]*Wind_std[y])
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

f = open('Optimal_values02', "a") #Makes (or add to the existing) 'Optimal_values02'
h = open('Gamma_values02', "a") #Makes (or add to the existing) 'Gamma_values02'
i = open('Capacity_values02', "a") #Makes (or add to the existing) 'Capacity_values02'

std_value = []; cf_value = []; gamma_value = []; capacity = []
plt.ion()
for x in range(250000): #Number of points we want to make
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
