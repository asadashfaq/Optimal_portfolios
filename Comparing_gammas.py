#!/usr/bin/python
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
Optimal_values = np.loadtxt('Optimal_values02') #Load the standard deviation and capacity factor
Gamma_values = np.loadtxt('Gamma_values02') #Load the gamma-values
Std = (Optimal_values[::2]) #Pick out the standard deviations from Optimal_values
Cf = (Optimal_values[1::2]) #Pick out the capacity factor from Optimal_values
Gamma=np.array([Gamma_values[30*x:(x+1)*30] for x in range(len(Std))]) #Converting the gamma-values in listform into 30-element arrays.
cf = list(Cf) #Make capacity factor into a list, so we can use index later (cause I have no idea how to do it with arrays ;) )

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
e = []; new_std = []; new_cf = []
for x in range(10):
	over= np.array(list((Std>147+4*x).nonzero())) #Picks standard deviation values above 147 + 4*x
	under = np.array(list((Std<151+4*x).nonzero())) #Picks standard deviation values below 151 + 4*x
	indices = np.intersect1d(over,under) #Finding the indices of the values that are both in "over" and "under"
	#a = list(Std[indices])
	list_cf = list(Cf[indices]) #making the capacity factors in the interval into a list
	top_values = ([cf.index(hq.nlargest(100,list_cf)[y]) for y in range(100)]) #Finds the top n values in the interval
	average_gamma = (np.average(Gamma[top_values], axis=0))#*Mean_load/Wind_cf
	e.append(average_gamma)
	new_std.append(total_std(average_gamma))
	new_cf.append(total_cf(average_gamma))
f = []
for x in range(30):
	for y in range(10):
		f.append(e[y][x]) #Making the gamma-values in a list, so that it can be plotted with plt.bar
my_colors = 'rrrrrrrrrrbbbbbbbbbb' #A very brilliant way of coloring every 10 bars red and blue
plt.bar(range(len(f)), f, width=0.5, bottom=0, color=my_colors)
ind = np.arange(30)*10 + 5
plt.xticks(ind, ('AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK'))
plt.show()
