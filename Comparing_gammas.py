#!/usr/bin/python
import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show

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

#This function makes random numbers, and then add a correction, beta, to match the constraint that gamma_eu = 1
def make_gamma():
	gamma_0 = np.random.random(30)*2
	beta = sum(gamma_0*Mean_load)/EU_L.mean() - 1
	gamma = gamma_0/(1+beta)
	return gamma

def make_gamma_1():
	gamma_1 = np.ones(30)
	beta = sum(gamma_1*Mean_load)/EU_L.mean() - 1
	gamma = gamma_1/(1+beta)
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
e = []; new_std = []; new_cf = []
#g = open('Useful_gammas', "a")
def find_optimal_gamma():
	for x in range(4):
		over= np.array(list((Std>147+8*x).nonzero())) #Picks standard deviation values above 147 + 4*x
		under = np.array(list((Std<155+8*x).nonzero())) #Picks standard deviation values below 151 + 4*x
		indices = np.intersect1d(over,under) #Finding the indices of the values that are both in "over" and "under"
		list_cf = list(Cf[indices]) #making the capacity factors in the interval into a list
		top_values = ([cf.index(hq.nlargest(500,list_cf)[y]) for y in range(500)]) #Finds the top n values in the interval
		average_gamma = (np.average(Gamma[top_values], axis=0))#*Mean_load/Wind_cf
		e.append(average_gamma)
		new_std.append(total_std(average_gamma))
		new_cf.append(total_cf(average_gamma))
	return e

e = find_optimal_gamma()
e.append(make_gamma_1())

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
	for x in range(100):
		Delta_EU.append(((x/100.00)*Wind_power + (1-x/100.00)*Solar_power) - Load_power)	
	return Delta_EU	
			
def balancing_energy(gammas):
	mismatch = mismatch_energy(gammas)
	negative_mismatch_index = [mismatch[y] < 0 for y in range(len(mismatch))]
	negative_mismatch = [negative_mismatch_index[x]*mismatch[x] for x in range(len(mismatch))]
	total_mismatch_energy = []
	for x in range(100):
		average = np.average(negative_mismatch[x])*(-1)
		total_mismatch_energy.append(np.sum(average))
	return total_mismatch_energy

def variability(gammas):
	mismatch = mismatch_energy(gammas)
	dDelta = []; variance = []
	for z in range(10):
		for x in [1,3,7,24,72,144]:
			for y in range(len(mismatch[0])):
				dDelta.append(mismatch[z][y]-mismatch[z][y-x])
	for x in range(60):
		variance.append(np.var(dDelta[70128*x:70128*(x+1)]))
	std = np.array([variance[x*6:(x+1)*6] for x in range(10)])
	return dDelta, std
titles = ['layout 1','layout 2','layout 3','layout 4','ref.point']
#shapes = ['ro','g<','bs','ys','ro','r>']
#for x in range(5):
#	a, b = variability(e[x])
#	steps = 231+x
#	plt.subplot(steps)	
#	plt.semilogy()		
#	plt.plot(b)
#	plt.legend(('1h','3h','7h','24h','72h','144h'))	
	#plt.axis([0,10,0,700])
#	plt.xlabel('alpha')
#	plt.ylabel('variance (GW^2)')
#	plt.title(titles[x])
#plt.show()
#plt.ion()
#titles = ['1h','3h','7h','24h','72h','144h']
#titles2 = ['layout 1','layout 2','layout 3','layout 4','ref.point']
#for y in range(5):
#	plt.figure(y)
#	for x in range(6):
#		steps = 231+x
#		plt.subplot(steps)
		#plt.axis([-400,400,0,30000])
#		plt.title(titles[x])
#		plt.xlabel('Mismatch power (GW)')
#		a, b = variability(e[y])
#		plt.hist(a[(x*70128+0*70128):((x+1)*70128+0*70128)],bins=100) #to use alpha = 0.8
#	plt.suptitle(titles2[y] + ' alpha = 0.0', fontsize=16)
#	plt.show()
shapes = ['ro','g<','b>','c-','ms','go','bo','co']
plt.ion()
for x in range(5):
	balancingenergy = balancing_energy(e[x])
	xcoord = np.linspace(0.01,1.00,100)
	plt.plot(xcoord,balancingenergy,shapes[x])
	plt.xlabel('alpha')
	plt.ylabel('absolute value of balancing energy')
	plt.legend(('layout 1','layout 2', 'layout 3', 'layout 4','ref.point','1','2','3'))

plt.show()
#my_colors = 'rrrrrrrrrrbbbbbbbbbb' #A very brilliant way of coloring every 10 bars red and blue
#plt.bar(range(len(f)), f, width=0.5, bottom=0, color=my_colors)
#ind = np.arange(30)*10 + 5
#plt.xticks(ind, ('AT','BA','BE','BG','CH','CZ','DE','DK','EE','ES','FI','FR','GB','GR','HR','HU','IE','IT','LT','LU','LV','NL','NO','PL','PT','RO','RS','SE','SI','SK'))
#plt.show()
